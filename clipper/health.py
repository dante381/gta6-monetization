"""Health monitor: notices silent wedges (stuck stage, GPU wait loops, error bursts, ...) and reports them.

    python -m clipper health [--watch] [--interval N] [--json]

`check_all` is read-only apart from one kv key (the worker stage signature). Every check is
individually guarded: a broken check can never crash the monitor. `run_once` de-duplicates by
finding key, writes data/logs/health.json, logs a `health` event, raises ATTENTION and notifies.

`watch` prints exactly one stdout line per new / resolved finding (a monitor consumes it):
    HEALTH <LEVEL> <code>: <message>        e.g. HEALTH WARN stuck_stage: ...
    HEALTH RESOLVED <code>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from clipper import heartbeat, log as elog, paths

logger = logging.getLogger("clipper.health")

LEVELS = {"info": 0, "warn": 1, "critical": 2}
IDLE_STAGES = ("loop", "start", "idle", "")
_LOG_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ (\w+) ([^:]*): (.*)$")

DEFAULTS = {
    "enabled": True, "interval_s": 60, "realert_s": 1800, "stuck_stage_s": 600, "gpu_wait_s": 120,
    "error_burst_n": 5, "error_burst_window_s": 300, "repeated_failure_n": 3,
    "repeated_failure_window_s": 1800, "backlog_n": 4, "backlog_age_s": 900, "disk_low_gb": 20,
    "no_selection_chunks": 8, "toast_min_gap_s": 60, "restart_cooldown_s": 600, "watch_interval_s": 30,
}


@dataclass
class Finding:
    level: str      # info | warn | critical
    code: str       # check name
    message: str
    key: str        # identifies the condition (de-dup)
    proc: str = ""  # process this concerns (the supervisor restarts the worker for some codes)

    def to_dict(self) -> dict:
        return asdict(self)


def hcfg(cfg, name: str):
    node = None
    try:
        node = cfg.get("health")
    except Exception:
        pass
    if node is not None:
        try:
            v = node.get(name)
            if v is not None:
                return v
        except Exception:
            pass
    return DEFAULTS[name]


# ---------------------------------------------------------------- readers
def _tail_text(path: Path, max_bytes: int = 400_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", "replace")
        return data if size <= max_bytes else data.split("\n", 1)[-1]
    except OSError:
        return ""


def _parse_ts(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


class Ctx:
    """Lazy, cached views over logs and events for one check_all pass."""

    def __init__(self, cfg, state, now: float):
        self.cfg, self.state, self.now = cfg, state, now
        self._events: list[dict] | None = None
        self._logs: dict[str, list[tuple[float, str, str, str]]] = {}

    def events(self) -> list[dict]:
        if self._events is None:
            out = []
            for line in _tail_text(paths.DATA.logs / "clips.jsonl", 1_000_000).splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
            self._events = out
        return self._events

    def log_lines(self, name: str) -> list[tuple[float, str, str, str]]:
        """(ts, level, logger, message); a bare 'Traceback' continuation line gets level TRACEBACK."""
        if name not in self._logs:
            out, last_ts = [], 0.0
            for line in _tail_text(paths.DATA.logs / name).splitlines():
                m = _LOG_RE.match(line)
                if m:
                    try:
                        last_ts = _parse_ts(m.group(1))
                    except ValueError:
                        continue
                    out.append((last_ts, m.group(2), m.group(3), m.group(4)))
                elif "Traceback" in line and last_ts:
                    out.append((last_ts, "TRACEBACK", "", line.strip()))
            self._logs[name] = out
        return self._logs[name]

    def all_log_names(self) -> list[str]:
        try:
            return sorted(p.name for p in paths.DATA.logs.glob("*.log"))
        except OSError:
            return []

    def stream(self) -> str | None:
        return self.state.current_stream()


def _norm(msg: str) -> str:
    return re.sub(r"\d+", "N", msg)[:160]


# ---------------------------------------------------------------- checks
def _worker_alive_hb(ctx: Ctx, name: str) -> dict | None:
    hb = heartbeat.read(name)
    if not hb:
        return None
    hang = float(ctx.cfg.supervisor.get("hang_s", 120))
    return hb if ctx.now - float(hb.get("ts", 0)) <= hang else None


def check_stuck_stage(ctx: Ctx) -> list[Finding]:
    hb = _worker_alive_hb(ctx, "worker")
    if not hb:
        return []
    sig = [hb.get("stage", ""), hb.get("chunk"), hb.get("cand"), hb.get("pid")]
    rec = ctx.state.kv_get("health.stage_sig")
    if not rec or rec.get("sig") != sig:
        ctx.state.kv_set("health.stage_sig", {"sig": sig, "since": ctx.now})
        return []
    if str(sig[0] or "") in IDLE_STAGES:
        return []
    stuck = ctx.now - float(rec["since"])
    if stuck <= float(hcfg(ctx.cfg, "stuck_stage_s")):
        return []
    what = f"stage={sig[0]}" + (f" chunk={sig[1]}" if sig[1] is not None else "") + \
        (f" cand={sig[2]}" if sig[2] is not None else "")
    return [Finding("critical", "stuck_stage",
                    f"worker alive but no progress for {stuck / 60:.0f} min ({what})",
                    f"stuck_stage:{sig[0]}:{sig[1]}:{sig[2]}:{sig[3]}", "worker")]


def check_gpu_wait_loop(ctx: Ctx) -> list[Finding]:
    win = float(hcfg(ctx.cfg, "gpu_wait_s"))
    out = []
    for name in ("worker.log", "recorder.log"):
        n = sum(1 for ts, _l, _lg, msg in ctx.log_lines(name) if "GPU unavailable" in msg and ctx.now - ts <= win)
        if n > 2:
            proc = name.split(".")[0]
            out.append(Finding("critical", "gpu_wait_loop",
                               f"{proc} logged 'GPU unavailable' {n}x in {win:.0f}s (likely an ffmpeg error "
                               f"mistaken for a GPU outage)", f"gpu_wait_loop:{proc}", proc))
    return out


def check_error_burst(ctx: Ctx) -> list[Finding]:
    win, need = float(hcfg(ctx.cfg, "error_burst_window_s")), int(hcfg(ctx.cfg, "error_burst_n"))
    out = []
    for name in ctx.all_log_names():
        msgs, prev_err = [], False
        for ts, lvl, _lg, msg in ctx.log_lines(name):
            err = lvl in ("ERROR", "CRITICAL")
            if (err or (lvl == "TRACEBACK" and not prev_err)) and ctx.now - ts <= win:
                msgs.append(msg)
            prev_err = err
        if len(msgs) >= need:
            top, cnt = collections.Counter(_norm(m) for m in msgs).most_common(1)[0]
            out.append(Finding("warn", "error_burst",
                               f"{len(msgs)} errors in {name} within {win:.0f}s; most common ({cnt}x): {top}",
                               f"error_burst:{name}", name.split(".")[0]))
    return out


def check_repeated_failure(ctx: Ctx) -> list[Finding]:
    win, need = float(hcfg(ctx.cfg, "repeated_failure_window_s")), int(hcfg(ctx.cfg, "repeated_failure_n"))
    c: collections.Counter = collections.Counter()
    for e in ctx.events():
        if e.get("kind") in ("clip_failed", "chunk_failed", "upload_failed") and e.get("error") \
                and ctx.now - float(e.get("ts", 0)) <= win:
            c[(e["kind"], str(e["error"]))] += 1
    out = []
    for (kind, err), n in c.items():
        if n >= need:
            h = hashlib.md5(err.encode()).hexdigest()[:8]
            out.append(Finding("warn", "repeated_failure", f"{kind} x{n}: {err[:200]}", f"repeated_failure:{kind}:{h}"))
    return out


def check_no_new_chunks(ctx: Ctx) -> list[Finding]:
    hb = _worker_alive_hb(ctx, "recorder")
    if not hb or hb.get("phase") != "recording":
        return []
    ref = ctx.state.kv_get("health.recording_since")
    if not ref or ref.get("pid") != hb.get("pid"):
        ref = {"pid": hb.get("pid"), "since": ctx.now}
        ctx.state.kv_set("health.recording_since", ref)
    last = float(ref["since"])
    sid = ctx.stream()
    if sid:
        c = ctx.state.get_chunk(sid, ctx.state.last_chunk_idx(sid))
        if c:
            last = max(last, float(c.recorded_at or 0))
    limit = 2 * float(ctx.cfg.recorder.get("chunk_seconds", 300))
    if ctx.now - last <= limit:
        return []
    return [Finding("critical", "no_new_chunks",
                    f"recorder is alive but no chunk registered for {(ctx.now - last) / 60:.0f} min "
                    f"(limit {limit / 60:.0f} min)", "no_new_chunks", "recorder")]


def check_backlog(ctx: Ctx) -> list[Finding]:
    sid = ctx.stream()
    if not sid:
        return []
    pend = ctx.state.chunks(sid, "recorded") + ctx.state.chunks(sid, "processing")
    if not pend:
        return []
    age = ctx.now - min(float(c.recorded_at or ctx.now) for c in pend)
    if len(pend) > int(hcfg(ctx.cfg, "backlog_n")) or age > float(hcfg(ctx.cfg, "backlog_age_s")):
        return [Finding("warn", "backlog", f"{len(pend)} chunks waiting, oldest {age / 60:.0f} min old", "backlog",
                        "worker")]
    return []


def check_process_down(ctx: Ctx) -> list[Finding]:
    sup = heartbeat.read("supervisor")
    hang = float(ctx.cfg.supervisor.get("hang_s", 120))
    if not sup or ctx.now - float(sup.get("ts", 0)) > max(hang, 3 * float(ctx.cfg.supervisor.get("heartbeat_s", 30))):
        return []
    grace = float(ctx.cfg.supervisor.get("startup_grace_s", hang))
    out = []
    for name, info in (sup.get("procs") or {}).items():
        if name == "chat":  # non-critical: covered by check_chat_down
            continue
        if info.get("status") != "running" or float(info.get("uptime_s", 0)) <= grace + hang:
            continue
        hb = heartbeat.read(name)
        age = None if not hb else ctx.now - float(hb.get("ts", 0))
        if age is None or age > 2 * hang:
            out.append(Finding("critical", "process_down",
                               f"{name} should be running but its heartbeat is "
                               f"{'missing' if age is None else f'{age:.0f}s old'}", f"process_down:{name}", name))
    return out


def _chat_enabled(ctx: Ctx) -> bool:
    try:
        return bool((ctx.cfg.get("chat") or {}).get("enabled", False))
    except Exception:
        return False


def check_chat_down(ctx: Ctx) -> list[Finding]:
    if not _chat_enabled(ctx):
        return []
    sup = heartbeat.read("supervisor")
    hang = float(ctx.cfg.supervisor.get("hang_s", 120))
    if not sup or ctx.now - float(sup.get("ts", 0)) > max(hang, 3 * float(ctx.cfg.supervisor.get("heartbeat_s", 30))):
        return []
    info = (sup.get("procs") or {}).get("chat")
    if not info:
        return []
    key = "chat_down"
    if info.get("status") in ("backoff", "capped", "stopped"):
        return [Finding("warn", "chat_down", f"chat collector is down (status {info.get('status')}); "
                        "chat signal will be missing", key, "chat")]
    if info.get("status") != "running":
        return []
    grace = float(ctx.cfg.supervisor.get("startup_grace_s", hang))
    if float(info.get("uptime_s", 0)) <= grace + hang:
        return []
    hb = heartbeat.read("chat")
    age = None if not hb else ctx.now - float(hb.get("ts", 0))
    if age is None or age > 2 * hang:
        return [Finding("warn", "chat_down", f"chat collector heartbeat is "
                        f"{'missing' if age is None else f'{age:.0f}s old'}; chat signal will be missing", key, "chat")]
    return []


def check_chat_silent(ctx: Ctx) -> list[Finding]:
    if not _chat_enabled(ctx):
        return []
    hb = _worker_alive_hb(ctx, "recorder")
    sid = ctx.stream()
    if not hb or hb.get("phase") != "recording" or not sid:
        return []
    wall = ctx.state.stream_wall_start(sid)
    if wall is None or ctx.now - wall <= 1800:
        return []
    if ctx.state.chat_total_msgs(sid) == 0:
        return [Finding("info", "chat_silent", "chat is empty or unreadable; chat signal is being ignored",
                        "chat_silent", "chat")]
    return []


def check_disk_low(ctx: Ctx) -> list[Finding]:
    free = shutil.disk_usage(paths.ROOT).free / 2**30
    lim = float(hcfg(ctx.cfg, "disk_low_gb"))
    if free < lim:
        return [Finding("critical" if free < lim / 4 else "warn", "disk_low",
                        f"only {free:.1f} GB free on the project drive (limit {lim:g} GB)", "disk_low")]
    return []


def check_memory(ctx: Ctx) -> list[Finding]:
    last = next((e for e in reversed(ctx.events()) if e.get("kind") == "job_memory"), None)
    if not last:
        return []
    peak, lim = float(last.get("peak_mb", 0)), float(last.get("limit_mb", 0))
    if lim > 0 and peak > 0.9 * lim:
        return [Finding("warn", "memory_near_cap", f"job memory peak {peak:.0f} MB is {peak / lim:.0%} of the "
                        f"{lim:.0f} MB cap", "memory_near_cap")]
    return []


def check_claude(ctx: Ctx) -> list[Finding]:
    out, win = [], 1800.0
    n = sum(1 for e in ctx.events() if e.get("kind") == "claude_unavailable" and ctx.now - float(e.get("ts", 0)) <= win)
    if n >= 3:
        out.append(Finding("warn", "claude_trouble", f"claude unavailable {n}x in the last 30 min",
                           "claude_trouble:unavailable"))
    lat = None
    for ts, _l, _lg, msg in ctx.log_lines("worker.log"):
        m = re.search(r"claude call ok in ([\d.]+)s", msg)
        if m and ctx.now - ts <= win:
            lat = float(m.group(1))
    if lat is not None and lat > 180:
        out.append(Finding("warn", "claude_trouble", f"latest claude call took {lat:.0f}s (>180s)",
                           "claude_trouble:slow"))
    return out


def check_no_selection(ctx: Ctx) -> list[Finding]:
    n = int(hcfg(ctx.cfg, "no_selection_chunks"))
    ev = [e for e in ctx.events() if e.get("kind") == "chunk_processed"][-n:]
    if len(ev) >= n and all(int(e.get("selected", 0)) == 0 for e in ev):
        return [Finding("info", "no_selection_streak",
                        f"last {n} chunks produced no selected clips; scoring may need calibration",
                        "no_selection_streak")]
    return []


def _attention_since(since: float) -> list[str]:
    out = []
    for line in _tail_text(paths.DATA.logs / "ATTENTION.txt", 100_000).splitlines():
        try:
            ts = _parse_ts(line[:19])
        except ValueError:
            continue
        if ts > since and "HEALTH " not in line[19:30]:
            out.append(line[19:].strip())
    return out


def check_publish_state(ctx: Ctx) -> list[Finding]:
    out, st = [], ctx.state
    if st.publish_hold():
        out.append(Finding("warn", "publish_state", f"publishing is on hold: {st.kv_get('hold.publish_reason') or '?'}",
                           "publish_hold"))
    for status in ("needs_review", "failed"):
        n = len(st.queue_items(status=status))
        if n:
            out.append(Finding("warn", "publish_state", f"{n} queue item(s) in status {status}", f"queue:{status}"))
    from clipper import config as _config
    if _config.paused(ctx.cfg):
        out.append(Finding("info", "publish_state", "PAUSE file present: publishing is paused", "paused"))
    since = float(st.kv_get("health.attn_seen") or ctx.now - 3600)
    lines = _attention_since(since)
    if lines:
        out.append(Finding("info", "attention", f"{len(lines)} new ATTENTION line(s); latest: {lines[-1][:200]}",
                           f"attention:{hashlib.md5(lines[-1].encode()).hexdigest()[:8]}"))
    return out


CHECKS: list[Callable[[Ctx], list[Finding]]] = [
    check_stuck_stage, check_gpu_wait_loop, check_error_burst, check_repeated_failure, check_no_new_chunks,
    check_backlog, check_process_down, check_disk_low, check_memory, check_claude, check_no_selection,
    check_publish_state, check_chat_down, check_chat_silent,
]


def check_all(cfg, state, now: float | None = None) -> list[Finding]:
    now = time.time() if now is None else now
    ctx = Ctx(cfg, state, now)
    out: list[Finding] = []
    for chk in list(CHECKS):
        try:
            out.extend(chk(ctx))
        except Exception as e:  # noqa: BLE001 - a broken check must never crash the monitor
            logger.warning("health check %s failed: %s", getattr(chk, "__name__", chk), e)
    return out


# ---------------------------------------------------------------- status file
def overall(findings: Sequence[Finding]) -> str:
    return max((f.level for f in findings), key=lambda l: LEVELS.get(l, 0), default="ok")


def write_status(findings: Sequence[Finding], now: float | None = None) -> Path:
    now = time.time() if now is None else now
    p = paths.DATA.logs / "health.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"ts": now, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
           "level": overall(findings), "findings": [f.to_dict() for f in findings]}
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for _ in range(5):
        try:
            os.replace(tmp, p)
            return p
        except PermissionError:
            time.sleep(0.05)
    tmp.unlink(missing_ok=True)
    return p


# ---------------------------------------------------------------- run / watch
def _cycle(cfg, state, notifier: Callable[[Finding], None] | None, now: float | None = None):
    """-> (current findings, new findings, resolved [(code, key)])."""
    now = time.time() if now is None else now
    findings = check_all(cfg, state, now)
    realert = float(hcfg(cfg, "realert_s"))
    alerts: dict = state.kv_get("health.alerts") or {}
    new, current_keys = [], set()
    for f in findings:
        current_keys.add(f.key)
        rec = alerts.get(f.key)
        if rec is None or now - float(rec.get("last", 0)) >= realert:
            new.append(f)
            alerts[f.key] = {"last": now, "code": f.code, "level": f.level}
    resolved = [(r["code"], k) for k, r in list(alerts.items()) if k not in current_keys]
    for _c, k in resolved:
        del alerts[k]
    state.kv_set("health.alerts", alerts)
    state.kv_set("health.attn_seen", now)
    try:
        write_status(findings, now)
    except Exception:
        logger.exception("could not write health.json")
    for f in new:
        elog.event("health", state="new", **f.to_dict())
        if f.level in ("warn", "critical"):
            elog.attention(f"HEALTH {f.level.upper()} {f.code}: {f.message}")
        if notifier is not None:
            try:
                notifier(f)
            except Exception:
                logger.exception("notifier failed")
    for code, key in resolved:
        elog.event("health", state="resolved", code=code, key=key)
    return findings, new, resolved


def run_once(cfg, state, notifier: Callable[[Finding], None] | None = None) -> list[Finding]:
    """One monitoring pass. Returns the NEWLY reported findings (de-duplicated by key)."""
    return _cycle(cfg, state, notifier)[1]


def _line(f: Finding) -> str:
    return f"HEALTH {f.level.upper()} {f.code}: {' '.join(f.message.split())}"


def watch(cfg, state, interval_s: float = 30, *, notifier=None, max_iterations: int | None = None,
          sleep: Callable[[float], None] = time.sleep, out=None) -> None:
    out = out or sys.stdout
    if notifier is None:
        from clipper import notify
        notifier = notify.make_notifier(cfg)
    i = 0
    while max_iterations is None or i < max_iterations:
        i += 1
        try:
            _f, new, resolved = _cycle(cfg, state, notifier)
            for f in new:
                print(_line(f), file=out, flush=True)
            for code, _key in resolved:
                if code != "attention":
                    print(f"HEALTH RESOLVED {code}", file=out, flush=True)
        except Exception:
            logger.exception("health pass failed")
        if max_iterations is None or i < max_iterations:
            sleep(interval_s)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clipper health")
    ap.add_argument("--watch", action="store_true", help="run forever, one stdout line per new/resolved finding")
    ap.add_argument("--interval", type=float, default=None, help="seconds between passes with --watch")
    ap.add_argument("--json", action="store_true", help="print findings as JSON")
    args = ap.parse_args(list(argv) if argv is not None else None)
    from clipper.config import load_config
    from clipper.state import StateDB
    cfg = load_config()
    elog.setup_logging("health")
    state = StateDB()
    try:
        if args.watch:
            try:
                watch(cfg, state, args.interval or float(hcfg(cfg, "watch_interval_s")))
            except KeyboardInterrupt:
                pass
            return 0
        findings = check_all(cfg, state)
        if args.json:
            print(json.dumps({"level": overall(findings), "findings": [f.to_dict() for f in findings]}, indent=2))
        elif not findings:
            print("OK: no findings")
        else:
            for f in sorted(findings, key=lambda x: -LEVELS[x.level]):
                print(f"{f.level.upper():9} {f.code:20} {f.message}")
        lv = overall(findings)
        return 1 if lv == "critical" else 2 if lv == "warn" else 0
    finally:
        state.close()


if __name__ == "__main__":
    raise SystemExit(main())
