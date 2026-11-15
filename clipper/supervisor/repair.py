"""Claude CLI repair agent, launched by the supervisor when the same worker crash repeats.

The agent is NOT trusted. Its work is checked here, outside the agent:
  1. no protected path touched (git diff/status, plus a snapshot of the gitignored protected
     files git can't see: secrets/, client_secret.json, PAUSE, tools/, .venv packages),
  2. HEAD not moved (the agent may not commit),
  3. the unit test suite passes, then the integration test if a sample chunk exists.
Pass: only the changed paths are committed as "repair: <signature>". Fail: everything goes back
to the recorded HEAD (plus any uncommitted user edits that were there before), new untracked
files under clipper/ and tests/ are removed, and nothing else on disk is deleted.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets as _secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from clipper import claude_cli, log as elog, paths
from clipper.env import subprocess_env
from clipper.ffmpeg import kill_tree

log = logging.getLogger(__name__)

Runner = Callable[[Sequence[str], Path, float], tuple[int, str]]

CO_AUTHOR = "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"

# Repo-relative paths the agent must never change (exact files, or prefixes ending in "/").
PROTECTED = (
    "config/locked.yaml", "secrets/", "client_secret.json", "clipper/supervisor/", "requirements.txt",
    ".gitignore", "tools/", ".venv/", ".git/",
    # the supervisor's own checks: the agent may not weaken the test harness or its own prompt
    "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "tests/conftest.py", "tests/fakes/",
    "prompts/repair.md", "conftest.py", "sitecustomize.py",
    # safety-critical code: thresholds/caps/kill switch, paths, env, subprocess helpers, publishing, crash/state
    "clipper/config.py", "clipper/paths.py", "clipper/env.py", "clipper/ffmpeg.py", "clipper/claude_cli.py",
    "clipper/publish/", "clipper/crash.py", "clipper/state.py",
)
# Protected by file name anywhere in the tree (test-gate / interpreter start-up hooks).
PROTECTED_NAMES = ("conftest.py", "sitecustomize.py", "usercustomize.py")
# Paths the agent may not even read or search (credentials, git internals).
SECRET_PATHS = ("secrets/", "client_secret.json", ".git/")
# New untracked files left by a failed attempt are deleted anywhere EXCEPT under these.
NEVER_CLEAN = ("data/", "music/", "secrets/", "client_secret.json", "tools/", ".venv/", ".tmp/", "hud_refs/")
# Roots whose every file is content-hashed before/after the agent runs (git can't see them).
DEEP_ROOTS = (".git/hooks", ".git/config", "tools", ".venv")
DEEP_HASH_FULL_MAX = 4_000_000       # files up to this size are fully sha256'd
DEEP_BACKUP_FILE_MAX = 1_000_000     # files up to this size are kept in memory for restore
DEEP_BACKUP_BUDGET = 64_000_000      # total in-memory backup budget outside .git

ALLOWED_TOOLS = ("Read Edit Write Grep Glob Bash(git status) Bash(git diff --stat) "
                 "Bash(git log --oneline -n 5)")


def _rules(tool: str, paths_: Sequence[str], root: Path) -> list[str]:
    """Deny rules for one tool, in relative and project-root-absolute (forward slash, both drive-letter cases) forms."""
    rr = root.as_posix().rstrip("/")
    roots = {rr}
    if len(rr) > 1 and rr[1] == ":":
        roots.add(rr[0].lower() + rr[1:])   # the other drive-letter case
    out: list[str] = []
    for p in paths_:
        pat = p + "**" if p.endswith("/") else p
        out.append(f"{tool}(./{pat})")
        for r_ in sorted(roots):
            out.append(f"{tool}({r_}/{pat})")
    return out


def disallowed_tools(root: Path | str = paths.ROOT) -> str:
    root = Path(root)
    rules: list[str] = []
    for tool in ("Edit", "Write"):
        rules += _rules(tool, PROTECTED, root)
    for tool in ("Read", "Grep", "Glob"):
        rules += _rules(tool, SECRET_PATHS, root)
    rules += ["Bash(pip*)", "Bash(winget*)", "WebFetch", "WebSearch"]
    seen: set[str] = set()
    return " ".join(r for r in rules if not (r in seen or seen.add(r)))


DISALLOWED_TOOLS = disallowed_tools()

# ---------------------------------------------------------------- untrusted text handling
CR, LF = chr(13), chr(10)
CRLF = CR + LF
_CTRL = re.compile("[" + "".join(re.escape(chr(c)) for c in [*range(0, 9), 11, 12, *range(14, 32), *range(127, 160)]) + "]")
_SECRET_RES = (
    re.compile(r"ya29\.[\w\-.]+"), re.compile(r"1//[\w\-]{10,}"), re.compile(r"AIza[\w\-]{10,}"),
    re.compile(r"GOCSPX-[\w\-]+"),
    re.compile(r"(?i)(client_secret|refresh_token|access_token|id_token)[\"']?\s*[:=]\s*[\"']?[^\s\"',}]+"),
    re.compile(r"(?i)client_secret"),
    re.compile(r"[A-Za-z0-9_\-+/=]{40,}"),
)


def redact(text: str) -> str:
    for rx in _SECRET_RES:
        text = rx.sub("[REDACTED]", text)
    return text


def sanitize(text: object, limit: int, tail: bool = True) -> str:
    """Untrusted text for the prompt/logs: control chars stripped, secrets redacted, truncated."""
    t = str(text if text is not None else "").replace(CRLF, LF).replace(CR, LF)
    t = _CTRL.sub("", t)
    t = redact(t).replace("{{", "{ {").replace("}}", "} }")
    if len(t) > limit:
        t = ("[...truncated...]" + LF + t[-limit:]) if tail else (t[:limit] + LF + "[...truncated...]")
    return t


def is_protected(rel: str) -> bool:
    low = rel.replace("\\", "/").lower()
    while low.startswith("./"):
        low = low[2:]
    if low.rsplit("/", 1)[-1] in PROTECTED_NAMES or low.endswith(".pth"):
        return True
    for p in PROTECTED:
        if p.endswith("/"):
            if low.startswith(p) or low == p[:-1]:
                return True
        elif low == p:
            return True
    return False


def _never_clean(rel: str) -> bool:
    low = rel.replace("\\", "/").lower()
    return any(low.startswith(p) or low == p.rstrip("/") for p in NEVER_CLEAN)


_TEST_BAD_RES = (
    re.compile(r"pytest\.mark\.(skip|skipif|xfail|network|claude)\b"), re.compile(r"\bxfail\b"),
    re.compile(r"\bskipif\b"), re.compile(r"pytest\.(skip|xfail|importorskip)\s*\("),
    re.compile(r"""['"]-[km]['"]"""), re.compile(r"--deselect|--ignore|collect_ignore|pytest_collection_modifyitems"),
    re.compile(r"(?<![\w-])-[km]\s+\S"),
)


def gate_hits(text: str) -> list[str]:
    return sorted({m.group(0) for rx in _TEST_BAD_RES for m in rx.finditer(text)})


def claude_args(cfg, repo_root: Path | str = paths.ROOT) -> list[str]:
    return ["-p", "--model", cfg.claude.model, "--effort", cfg.claude.effort_repair,
            "--no-session-persistence", "--safe-mode", "--permission-mode", "dontAsk",
            "--tools", "Read,Edit,Write,Grep,Glob,Bash",
            "--allowedTools", ALLOWED_TOOLS, "--disallowedTools", disallowed_tools(repo_root)]


def default_runner(cmd: Sequence[str], cwd: Path, timeout: float) -> tuple[int, str]:
    """Run a command, merged stdout+stderr, killing the whole tree on timeout."""
    proc = subprocess.Popen(list(map(str, cmd)), cwd=str(cwd), env=subprocess_env(), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out, _ = proc.communicate()
        return 124, (out or b"").decode("utf-8", "replace") + f"\n[timed out after {timeout}s]"
    return proc.returncode, (out or b"").decode("utf-8", "replace")


# ---------------------------------------------------------------- hold flags
PRIOR_KEY = "repair.prior_hold"
BLOCK_KEY = "repair.blocked"


def hold_for_repair(state) -> None:
    """Pause publishing and keep chunks. An existing hold (e.g. 'auth') is preserved."""
    reason = state.kv_get("hold.publish_reason", "") or ""
    if state.kv_get(PRIOR_KEY) is None:  # remember what the user/supervisor had before we touched anything
        state.kv_set(PRIOR_KEY, {"hold": bool(state.publish_hold()), "reason": reason,
                                 "keep": bool(state.keep_chunks())})
    reasons = [r for r in reason.split("+") if r]
    if state.publish_hold() and reasons and "repair" not in reasons:
        state.set_publish_hold(True, f"{reason}+repair")
    elif not (state.publish_hold() and "repair" in reasons):
        state.set_publish_hold(True, "repair")
    state.set_keep_chunks(True)


def release_repair_hold(state) -> None:
    """After a successful repair: restore exactly the hold / keep-chunks state recorded before the
    repair. A hold whose reason does not include 'repair' is never released."""
    prior = state.kv_get(PRIOR_KEY) or {}
    reasons = [r for r in (state.kv_get("hold.publish_reason", "") or "").split("+") if r]
    if state.publish_hold() and "repair" in reasons:
        rest = [r for r in reasons if r != "repair"]
        if rest:                       # something else (e.g. 'auth') joined meanwhile: it stays
            state.set_publish_hold(True, "+".join(rest))
        else:
            state.set_publish_hold(bool(prior.get("hold", False)), prior.get("reason", "") or "")
        state.set_keep_chunks(bool(prior.get("keep", False)))
    state.kv_set(PRIOR_KEY, None)


# ---------------------------------------------------------------- agent
class RepairError(RuntimeError):
    pass


class RepairAgent:
    def __init__(self, cfg, state, repo_root: Path | str = paths.ROOT, runner: Runner | None = None,
                 job=None, test_cmd: Sequence[str] | None = None):
        self.cfg = cfg
        self.state = state
        self.repo_root = Path(repo_root).resolve()
        self.runner: Runner = runner or default_runner
        self.job = job  # JobObject: claude/git/pytest started during a repair are put inside it
        scfg = cfg.supervisor
        self.max_attempts = int(scfg.get("max_repairs_per_stream", 3))
        self.timeout_s = float(scfg.get("repair_timeout_s", 1800))
        self.test_timeout_s = float(scfg.get("repair_test_timeout_s", 900))
        self.test_cmd = list(test_cmd or [str(paths.VENV_PY), "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                                          "-m", "not network and not claude"])

    # -------------------------------------------------------------- budget
    def _key(self) -> str:
        return f"repair.attempts.{self.state.current_stream() or 'none'}"

    def attempts_used(self) -> int:
        return int(self.state.kv_get(self._key(), 0) or 0)

    def exhausted(self) -> bool:
        return self.attempts_used() >= self.max_attempts

    def _attention_once(self, signature: str) -> None:
        k = f"repair.attention.{self.state.current_stream() or 'none'}"
        if self.state.kv_get(k, False):
            return
        self.state.kv_set(k, True)
        elog.attention(f"Repair agent used all {self.max_attempts} attempts for this stream; the worker keeps "
                       f"crashing ({signature}). Recording continues, publishing stays paused. Fix it by hand, "
                       f"then clear the hold.", signature=signature)

    # -------------------------------------------------------------- git helpers
    def _git(self, *args: str, timeout: float = 120, check: bool = True) -> str:
        # command-line -c beats .git/config: no hooks, no fsmonitor, no external diff/pager from the work tree
        rc, out = self.runner(["git", "-c", "core.quotepath=off", "-c", "core.fsmonitor=false",
                               "-c", f"core.hooksPath={os.devnull}", "-c", "core.pager=cat", *args],
                              self.repo_root, timeout)
        if check and rc != 0:
            raise RepairError(f"git {' '.join(args[:3])} failed (rc={rc}): {out.strip()[-500:]}")
        return out

    def _check_repo(self) -> None:
        top = self._git("rev-parse", "--show-toplevel").strip()
        if os.path.normcase(str(Path(top).resolve())) != os.path.normcase(str(self.repo_root)):
            raise RepairError(f"{self.repo_root} is not a git work tree root (toplevel={top!r})")

    def _untracked(self) -> set[str]:
        out = self._git("ls-files", "--others", "--exclude-standard", "-z")
        return {p for p in out.split("\0") if p}

    def _changed_since(self, base: str, head: str, untracked_before: set[str]) -> set[str]:
        changed: set[str] = set()
        # base commit vs work tree (tracked files); --no-renames so a moved file shows both paths
        out = self._git("diff", "--no-renames", "--no-ext-diff", "--name-only", "-z", base)
        changed |= {p for p in out.split("\0") if p}
        cur_head = self._git("rev-parse", "HEAD").strip()
        if cur_head != head:
            out = self._git("diff", "--no-renames", "--no-ext-diff", "--name-only", "-z", head, cur_head, check=False)
            changed |= {p for p in out.split("\0") if p}
        changed |= self._untracked() - untracked_before
        return changed

    # -------------------------------------------------------------- snapshot of files git can't see
    def _snapshot(self) -> dict[str, object]:
        r = self.repo_root
        snap: dict[str, object] = {}

        def add_file(p: Path):
            rel = p.relative_to(r).as_posix()
            try:
                data = p.read_bytes() if p.stat().st_size <= 2_000_000 else None
                snap[rel] = ("file", hashlib.sha256(data).hexdigest() if data is not None else p.stat().st_size,
                             data if (rel.startswith("secrets/") or rel == "client_secret.json") else None)
            except OSError:
                pass

        for name in ("config/locked.yaml", "client_secret.json", "requirements.txt", ".gitignore", "pytest.ini",
                     "tests/conftest.py", "prompts/repair.md"):
            if (r / name).is_file():
                add_file(r / name)
        for d in ("secrets", "clipper/supervisor", "tests/fakes"):
            if (r / d).is_dir():
                for p in sorted((r / d).rglob("*")):
                    if p.is_file() and "__pycache__" not in p.parts:
                        add_file(p)
        snap["PAUSE"] = ("exists", (r / "PAUSE").exists(), None)
        self._snapshot_deep(snap)
        return snap

    def _snapshot_deep(self, snap: dict) -> None:
        """Recursive content snapshot of .git/hooks, .git/config, tools/ and .venv/ (no __pycache__):
        size + mtime + sha256 (whole file up to DEEP_HASH_FULL_MAX, else first/last 64 KB), plus the
        bytes of small files so a change can be undone."""
        r = self.repo_root
        budget = [DEEP_BACKUP_BUDGET]

        def add(path: str, rel: str):
            try:
                st = os.lstat(path)
                if os.path.islink(path):
                    snap[rel] = ("link", os.readlink(path), None)
                    return
                h = hashlib.sha256()
                data = None
                with open(path, "rb") as f:
                    if st.st_size <= DEEP_HASH_FULL_MAX:
                        blob = f.read()
                        h.update(blob)
                        if st.st_size <= DEEP_BACKUP_FILE_MAX and (rel.startswith(".git/") or budget[0] > 0):
                            data = blob
                            if not rel.startswith(".git/"):
                                budget[0] -= st.st_size
                    else:
                        h.update(f.read(65536))
                        f.seek(max(0, st.st_size - 65536))
                        h.update(f.read(65536))
                snap[rel] = ("deep", (st.st_size, st.st_mtime_ns, h.hexdigest()), data)
            except OSError:
                snap[rel] = ("deep", ("unreadable",), None)

        for root in DEEP_ROOTS:
            base = r / root
            if base.is_file() or base.is_symlink():
                add(str(base), root)
            elif base.is_dir():
                snap[root + "/"] = ("dir", True, None)
                for dp, dns, fns in os.walk(base):
                    dns[:] = sorted(d for d in dns if d != "__pycache__")
                    for d in dns:
                        full = os.path.join(dp, d)
                        if os.path.islink(full):
                            snap[Path(full).relative_to(r).as_posix()] = ("link", os.readlink(full), None)
                    for fn in sorted(fns):
                        full = os.path.join(dp, fn)
                        add(full, Path(full).relative_to(r).as_posix())
            else:
                snap[root] = ("absent", True, None)

    def _snapshot_violations(self, before: dict, after: dict) -> list[str]:
        bad = []
        for k in set(before) | set(after):
            b, a = before.get(k), after.get(k)
            if (b[:2] if b else None) != (a[:2] if a else None):
                bad.append(k)
        return sorted(bad)

    def _restore_snapshot(self, before: dict, after: dict) -> list[str]:
        """Undo what the snapshots show. Returns the keys that could NOT be restored."""
        r = self.repo_root
        lost: list[str] = []
        for k, v in before.items():
            kind, _, data = v
            changed = after.get(k, (None, None))[:2] != v[:2]
            if kind in ("file", "deep") and changed:
                if data is not None:
                    try:
                        (r / k).parent.mkdir(parents=True, exist_ok=True)
                        (r / k).write_bytes(data)
                    except OSError:
                        lost.append(k)
                elif kind == "deep":
                    lost.append(k)
            elif kind == "link" and changed:
                lost.append(k)
            elif kind == "exists":
                p = r / k
                if v[1] and not p.exists():
                    p.write_text("", encoding="utf-8")
                elif not v[1] and p.exists():
                    p.unlink()
        for k, v in after.items():  # files the agent created in protected places
            if k in before:
                continue
            if v[0] == "deep" or (v[0] == "file" and (k.startswith("secrets/") or k == "client_secret.json")):
                try:
                    (r / k).unlink(missing_ok=True)
                except OSError:
                    lost.append(k)
            elif v[0] == "link":
                lost.append(k)
        return sorted(set(lost))

    # -------------------------------------------------------------- revert / commit
    def _revert(self, head: str, base: str, untracked_before: set[str]) -> list[str]:
        self._git("reset", "--hard", "-q", head)
        if base != head:  # put the user's own uncommitted edits back
            self._git("stash", "apply", "-q", base, check=False)
        left: list[str] = []
        for rel in sorted(self._untracked() - untracked_before):
            if not _never_clean(rel):
                p = self.repo_root / rel
                try:
                    p.unlink()
                except OSError:
                    left.append(rel)
                    continue
                d = p.parent
                while d != self.repo_root and d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
                    d = d.parent
            else:
                left.append(rel)
        if left:
            log.warning("repair revert left new untracked files in place: %s", left)
        return left

    def _commit(self, signature: str, changed: set[str]) -> str:
        files = sorted(changed)
        self._git("--literal-pathspecs", "add", "-A", "--", *files)
        ident: list[str] = []
        if not self._git("config", "user.email", check=False).strip():
            ident = ["-c", "user.name=GTA6 Clipper repair", "-c", "user.email=repair@localhost"]
        sig = " ".join(signature.split())[:120]
        # `git commit -- <paths>` commits only the agent's paths, never anything else in the index
        self._git(*ident, "--literal-pathspecs", "commit", "-q", "-m", f"repair: {sig}", "-m", CO_AUTHOR,
                  "--", *files)
        return self._git("rev-parse", "HEAD").strip()

    # -------------------------------------------------------------- prompt
    def build_prompt(self, report: dict, log_tail: str) -> str:
        tpl = (paths.PROMPTS / "repair.md").read_text(encoding="utf-8")
        rep = "\n".join(f"{sanitize(k, 40)}: {sanitize(v, 500, tail=False)}"
                        for k, v in report.items() if k != "traceback")
        fields = {
            "SIGNATURE": sanitize(report.get("signature", "?"), 300, tail=False).replace("\n", " "),
            "REPORT": rep,
            "TRACEBACK": sanitize(report.get("traceback", "") or "(none)", 6000),
            "LOG_TAIL": sanitize(log_tail or "(empty)", 8000),
            "PROTECTED": "\n".join(f"- {p}" for p in PROTECTED),
            "TEST_CMD": " ".join(f'"{c}"' if " " in c else c for c in self.test_cmd),
            "REPO": str(self.repo_root),
            "NONCE": _secrets.token_hex(6),
        }
        # one pass, so placeholder-looking text inside the untrusted fields is never expanded
        return re.sub(r"\{\{(\w+)\}\}", lambda m: fields.get(m.group(1), m.group(0)), tpl)

    # -------------------------------------------------------------- the attempt
    def attempt(self, report: dict, log_tail: str) -> bool:
        signature = str(report.get("signature") or "unknown")
        if self.state.kv_get(BLOCK_KEY):
            hold_for_repair(self.state)
            elog.event("repair", signature=signature, ok=False, reason="blocked_unrecoverable_changes",
                       paths=self.state.kv_get(BLOCK_KEY))
            return False
        if self.exhausted():
            hold_for_repair(self.state)
            self._attention_once(signature)
            elog.event("repair", signature=signature, ok=False, reason="cap_reached",
                       attempts=self.attempts_used())
            return False
        n = self.attempts_used() + 1
        self.state.kv_set(self._key(), n)
        hold_for_repair(self.state)
        t0 = time.monotonic()
        stop_adopt = threading.Event()
        adopter = None
        if self.job is not None:
            def adopt_loop():
                while not stop_adopt.wait(0.5):
                    try:
                        self.job.adopt_descendants()
                    except Exception:
                        pass
            adopter = threading.Thread(target=adopt_loop, name="repair-adopt", daemon=True)
            adopter.start()
        ok, reason, changed, commit = False, "", set(), None
        try:
            ok, reason, changed, commit = self._attempt(report, log_tail, signature)
        except Exception as e:  # never let the repair flow crash the supervisor
            log.exception("repair attempt failed")
            ok, reason = False, f"error: {e}"
        finally:
            stop_adopt.set()
            if adopter:
                adopter.join(timeout=2)
        elog.event("repair", signature=signature, ok=ok, reason=reason, attempt=n, max_attempts=self.max_attempts,
                   changed=sorted(changed), commit=commit, seconds=round(time.monotonic() - t0, 1))
        if not ok and self.exhausted():
            self._attention_once(signature)
        return ok

    def _never_commit(self, rel: str) -> bool:
        low = rel.replace("\\", "/").lower()
        return any(low.startswith(p) for p in ("data/", "music/", "hud_refs/", ".tmp/"))

    def _gate_violations(self, base: str, changed: set[str], untracked_before: set[str]) -> list[str]:
        """Protected paths, runtime-data paths, and the test gate (tests may only be added or modified,
        never deleted, skipped, xfailed or deselected)."""
        bad = [p for p in sorted(changed) if is_protected(p) or self._never_commit(p)]
        for p in sorted(changed):
            if p.replace("\\", "/").lower().startswith("tests/") and not (self.repo_root / p).exists():
                bad.append(f"{p} (test file deleted)")
        hits: list[str] = []
        diff = self._git("diff", "--no-renames", "--no-ext-diff", "--no-textconv", "-U0", base, "--", "tests",
                         check=False)
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                hits += gate_hits(line[1:])
        for p in sorted(self._untracked() - untracked_before):
            if p.replace("\\", "/").lower().startswith("tests/"):
                try:
                    fp = self.repo_root / p
                    if fp.is_file() and fp.stat().st_size <= 2_000_000:
                        hits += gate_hits(fp.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    pass
        if hits:
            bad.append(f"test gate weakened (skip/xfail/deselect): {sorted(set(hits))[:5]}")
        return bad

    def _block(self, lost: list[str]) -> None:
        self.state.kv_set(BLOCK_KEY, lost[:20])
        elog.attention("Repair agent changed files that could not be restored (" + ", ".join(lost[:5]) +
                       "). Automatic repair is disabled until you inspect .git/hooks, .git/config, tools/ and "
                       ".venv/ and clear the 'repair.blocked' state.", signature="repair-blocked")

    def _attempt(self, report: dict, log_tail: str, signature: str):
        self._check_repo()
        head = self._git("rev-parse", "HEAD").strip()
        # a dangling commit of any uncommitted user edits, so a revert can put them back
        base = self._git("-c", "user.name=clipper", "-c", "user.email=clipper@localhost",
                         "stash", "create", "repair baseline").strip() or head
        untracked_before = self._untracked()
        snap_before = self._snapshot()
        changed: set[str] = set()

        def fail(reason: str, snap_after: dict | None = None):
            # 1. undo changes to .git/hooks, .git/config, tools/, .venv/, secrets BEFORE any git call
            lost = self._restore_snapshot(snap_before, snap_after if snap_after is not None else self._snapshot())
            if lost:
                self._block(lost)
                reason += f" [could not restore {lost[:5]}; repairs disabled]"
                log.error("repair rejected (%s): %s", signature, redact(reason))
                return False, reason, changed, None
            try:
                changed.update(self._changed_since(base, head, untracked_before))
            except Exception:
                pass
            self._revert(head, base, untracked_before)
            log.warning("repair rejected (%s): %s", signature, redact(reason))
            return False, redact(reason), changed, None

        try:
            # 1. the agent
            prompt = self.build_prompt(report, log_tail)
            unavailable = None
            try:
                rc, out, err = claude_cli.run_raw(claude_args(self.cfg, self.repo_root), prompt, cwd=self.repo_root,
                                                  timeout_s=self.timeout_s)
                log.info("repair agent rc=%s: %s", rc, sanitize(out or err, 500))
            except claude_cli.ClaudeUnavailable as e:
                rc, unavailable = -1, str(e)

            # 2. what did it touch? Files git can't see first, with no git call before hooks/config are verified.
            snap_after = self._snapshot()
            bad_snap = self._snapshot_violations(snap_before, snap_after)
            if bad_snap:
                return fail(f"protected paths touched: {bad_snap[:20]}", snap_after)
            if unavailable:
                return fail(f"claude unavailable: {unavailable}", snap_after)
            changed |= self._changed_since(base, head, untracked_before)
            if self._git("rev-parse", "HEAD").strip() != head:
                return fail("agent moved HEAD (commits are not allowed)", snap_after)
            bad = self._gate_violations(base, changed, untracked_before)
            if bad:
                return fail(f"protected paths touched: {bad}", snap_after)
            if not changed:
                return fail(f"agent made no changes (rc={rc})", snap_after)

            # 3. tests, run by us from the repo root (the agent's own claim that tests pass is not trusted)
            trc, tout = self.runner(self.test_cmd, self.repo_root, self.test_timeout_s)
            if trc != 0:
                return fail(f"tests failed (rc={trc}): {sanitize(tout.strip(), 400)}")
            sample = paths.DATA.samples / "sample_chunk.ts"
            if sample.exists() and (self.repo_root / "tests" / "integration").is_dir():
                irc, iout = self.runner([*self.test_cmd[:3], "-x", "-q", "-p", "no:cacheprovider",
                                         "tests/integration"], self.repo_root, self.test_timeout_s)
                if irc != 0:
                    return fail(f"integration test failed (rc={irc}): {sanitize(iout.strip(), 400)}")
            # the test run itself must not have left protected changes either
            snap_after = self._snapshot()
            bad_snap = self._snapshot_violations(snap_before, snap_after)
            if bad_snap:
                return fail(f"protected paths changed during the test run: {bad_snap[:20]}", snap_after)
            changed |= self._changed_since(base, head, untracked_before)
            bad = self._gate_violations(base, changed, untracked_before)
            if bad:
                return fail(f"protected paths changed during the test run: {bad}", snap_after)

            # 4. keep it
            commit = self._commit(signature, changed)
        except Exception as e:
            log.exception("repair verification error; reverting")
            return fail(f"error: {e}")
        log.info("repair committed %s for %s: %s", commit[:10], signature, sorted(changed))
        return True, "committed", changed, commit
