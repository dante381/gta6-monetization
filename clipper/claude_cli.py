"""Headless Claude CLI (claude -p) wrapper used for all runtime AI calls.

Every call: model set explicitly (claude-sonnet-5-5), effort per task, JSON output validated
against a JSON schema, Read-only tools, no session persistence (nothing written to C:),
--safe-mode so the user's plugins/hooks/CLAUDE.md never leak into runtime calls.

Tests swap the binary with CLAUDE_BIN (e.g. "python tests/fakes/fake_claude.py").
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from . import paths
from .env import subprocess_env

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

UNAVAILABLE_PATTERNS = (
    "usage limit", "rate limit", "rate_limit", "limit reached", "quota", "overloaded", "529",
    "not logged in", "please run /login", "invalid api key", "authentication", "unauthorized", "401",
    "credit balance", "network", "econnreset", "etimedout", "enotfound", "fetch failed",
)


class ClaudeError(RuntimeError):
    """Call returned, but output was unusable (bad JSON, schema mismatch)."""


class ClaudeUnavailable(ClaudeError):
    """Usage limit / auth / network / timeout. Callers fall back to audio+motion scoring."""


def claude_bin() -> list[str]:
    override = os.environ.get("CLAUDE_BIN")
    if override:
        parts = shlex.split(override, posix=False)
        parts = [p.strip('"') for p in parts]
        if parts[0].lower().endswith(".py"):
            return [sys.executable, *parts]
        return parts
    return ["claude"]


def load_schema(name: str) -> dict:
    return json.loads((paths.SCHEMAS / f"{name}.json").read_text(encoding="utf-8"))


def load_prompt(name: str) -> str:
    return (paths.PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def _looks_unavailable(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in UNAVAILABLE_PATTERNS)


def _extract_json(text: str) -> Any:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # first {...} or [...] block
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ClaudeError(f"no JSON in Claude output: {text[:300]!r}")


def run_raw(args: Sequence[str], prompt: str, *, cwd: Path | None = None, timeout_s: float = 300,
            on_tick: Callable[[], None] | None = None, tick_s: float = 5.0,
            env_extra: dict[str, str] | None = None) -> tuple[int, str, str]:
    """Run claude with explicit args; prompt is sent on stdin. Returns (rc, stdout, stderr)."""
    cmd = [*claude_bin(), *map(str, args)]
    cwd = Path(cwd or paths.DATA.claude_work)
    cwd.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=subprocess_env(env_extra), creationflags=NO_WINDOW)
    out: list[bytes] = []
    err: list[bytes] = []

    def drain(s, sink):
        for b in iter(lambda: s.read(65536), b""):
            sink.append(b)
        s.close()

    ts = [threading.Thread(target=drain, args=(proc.stdout, out), daemon=True),
          threading.Thread(target=drain, args=(proc.stderr, err), daemon=True)]
    for t in ts:
        t.start()
    try:
        proc.stdin.write(prompt.encode("utf-8"))
    finally:
        proc.stdin.close()
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            proc.wait(timeout=tick_s)
            break
        except subprocess.TimeoutExpired:
            if on_tick:
                on_tick()
            if time.monotonic() > deadline:
                from .ffmpeg import kill_tree
                kill_tree(proc)
                raise ClaudeUnavailable(f"claude timed out after {timeout_s}s")
    for t in ts:
        t.join(timeout=5)
    return proc.returncode, b"".join(out).decode("utf-8", "replace"), b"".join(err).decode("utf-8", "replace")


def call(prompt: str, *, schema: dict | str | None = None, effort: str = "low",
         add_dirs: Sequence[Path | str] = (), model: str | None = None, timeout_s: float = 300,
         on_tick: Callable[[], None] | None = None, tools: str = "Read") -> Any:
    """One headless judgement call. Returns parsed JSON (dict/list) if schema given, else text.

    Images: put absolute file paths in the prompt and include their folder in add_dirs; Claude
    opens them with the Read tool.
    """
    if isinstance(schema, str):
        schema = load_schema(schema)
    args = ["-p", "--model", model or DEFAULT_MODEL, "--effort", effort,
            "--output-format", "json", "--no-session-persistence", "--safe-mode",
            "--permission-mode", "dontAsk", "--tools", tools, "--allowedTools", tools]
    for d in add_dirs:
        args += ["--add-dir", str(d)]
    if schema is not None:
        args += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
    t0 = time.monotonic()
    rc, out, err = run_raw(args, prompt, timeout_s=timeout_s, on_tick=on_tick)
    dt = time.monotonic() - t0
    envelope: dict | None = None
    try:
        envelope = json.loads(out) if out.strip() else None
    except json.JSONDecodeError:
        envelope = None
    if isinstance(envelope, list):  # some versions emit a list of messages; take the result one
        envelope = next((m for m in reversed(envelope) if isinstance(m, dict) and m.get("type") == "result"), None)
    if rc != 0 or envelope is None or envelope.get("is_error"):
        detail = (envelope or {}).get("result") or err or out
        if _looks_unavailable(str(detail)) or rc != 0:
            raise ClaudeUnavailable(f"claude rc={rc}: {str(detail)[:500]}")
        raise ClaudeError(f"claude error: {str(detail)[:500]}")
    log.info("claude call ok in %.1fs (effort=%s, cost=%s)", dt, effort, envelope.get("total_cost_usd"))
    if schema is None:
        return envelope.get("result", "")
    so = envelope.get("structured_output")
    if so is not None:
        return so
    return _extract_json(str(envelope.get("result", "")))
