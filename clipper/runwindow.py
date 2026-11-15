"""Launch-day run window: run ONE bounded window, then never again until the user re-arms it.

Configured (locked, so the repair agent can't change it) in `config/locked.yaml`:

    run_window:
      start: "2026-11-19T00:00:00+05:30"
      end:   "2026-11-20T00:00:00+05:30"
      drain_grace_min: 180     # after `end`: finish processing / uploads, then stop
      uploads: true            # uploads allowed ONLY from `start` to `end` + grace

Lifecycle:  before -> active -> draining -> over.  When the supervisor finishes a window run it
calls `mark_consumed()`; from then on `supervise --window` refuses to start (and the scheduled task
is deleted) until the user runs `python -m clipper window rearm`.

    python -m clipper window status
    python -m clipper window install [--url URL]    # one-time Task Scheduler entry for `start`
    python -m clipper window remove
    python -m clipper window rearm                  # allow the window to run again

Uploads outside the window (and any run without `--window`) keep following `locked.test_mode`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape

from . import paths

TASK_NAME = "GTA6ClipperLaunchDay"
BEFORE, ACTIVE, DRAINING, OVER, NONE = "before", "active", "draining", "over", "none"


class WindowError(RuntimeError):
    pass


# ------------------------------------------------------------------ config
def _parse(ts: str) -> datetime:
    d = datetime.fromisoformat(str(ts))
    if d.tzinfo is None:                       # naive = the laptop's local time
        d = d.astimezone()
    return d


def get_window(cfg: Any) -> dict | None:
    """{'start','end','grace_end' (aware datetimes), 'uploads', 'drain_grace_min'} or None."""
    try:
        w = cfg.locked.get("run_window") if hasattr(cfg.locked, "get") else None
    except Exception:  # noqa: BLE001
        w = None
    if not w or not w.get("start") or not w.get("end"):
        return None
    start, end = _parse(w["start"]), _parse(w["end"])
    if end <= start:
        raise WindowError("run_window.end must be after run_window.start")
    grace = int(w.get("drain_grace_min", 180))
    return {"start": start, "end": end, "grace_end": end + timedelta(minutes=grace),
            "uploads": bool(w.get("uploads", False)), "drain_grace_min": grace}


def status(cfg: Any, now: datetime | None = None) -> str:
    w = get_window(cfg)
    if w is None:
        return NONE
    now = now or datetime.now(timezone.utc)
    if now < w["start"]:
        return BEFORE
    if now < w["end"]:
        return ACTIVE
    if now < w["grace_end"]:
        return DRAINING
    return OVER


WINDOW_ENV = "CLIPPER_WINDOW_RUN"


def uploads_allowed(cfg: Any, now: datetime | None = None) -> bool:
    """True only while a window with `uploads: true` is active or draining (never before/after),
    AND only inside processes started by `supervise --window` (it sets CLIPPER_WINDOW_RUN=1 for its
    children). Tests and manual runs never have the marker, so they can never upload because of it."""
    if os.environ.get(WINDOW_ENV) != "1":
        return False
    try:
        w = get_window(cfg)
    except Exception:  # noqa: BLE001
        return False
    if w is None or not w["uploads"]:
        return False
    return status(cfg, now) in (ACTIVE, DRAINING)


# ------------------------------------------------------------------ consumed flag
def state_path() -> Path:
    return paths.DATA.root / "run_window.json"


def read_state() -> dict:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return {}


def _write_state(d: dict) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def is_consumed() -> bool:
    return bool(read_state().get("consumed"))


def mark_started() -> None:
    d = read_state()
    d.setdefault("first_started_at", time.time())
    d["last_started_at"] = time.time()
    d["runs"] = int(d.get("runs", 0)) + 1
    _write_state(d)


def mark_consumed(reason: str = "window finished") -> None:
    d = read_state()
    d.update(consumed=True, consumed_at=time.time(), consumed_reason=reason)
    _write_state(d)


def rearm() -> None:
    d = read_state()
    d.update(consumed=False, rearmed_at=time.time())
    d.pop("consumed_at", None)
    d.pop("consumed_reason", None)
    _write_state(d)


def may_start(cfg: Any, now: datetime | None = None) -> tuple[bool, str]:
    """May a `supervise --window` run start now? (ok, human reason)."""
    w = get_window(cfg)
    if w is None:
        return False, "no run_window is configured in config/locked.yaml"
    if is_consumed():
        return False, "the run window was already used; run `python -m clipper window rearm` to allow it again"
    st = status(cfg, now)
    if st == BEFORE:
        return False, f"too early: the window starts at {w['start'].isoformat()}"
    if st == OVER:
        return False, f"the window ended at {w['end'].isoformat()}"
    return True, st


def seconds_until(cfg: Any, which: str, now: datetime | None = None) -> float:
    w = get_window(cfg)
    if w is None:
        return float("inf")
    now = now or datetime.now(timezone.utc)
    return (w[which] - now).total_seconds()


# ------------------------------------------------------------------ Task Scheduler (one-time task)
def supervise_args(url: str) -> list[str]:
    return ["-m", "clipper", "supervise", "--window", "--url", url]


def _current_user() -> str:
    import getpass
    dom = os.environ.get("USERDOMAIN")
    name = os.environ.get("USERNAME") or getpass.getuser()
    return f"{dom}\\{name}" if dom else name


def _fmt(d: datetime) -> str:
    return d.isoformat(timespec="seconds")


def task_xml(cfg: Any, url: str, user: str | None = None) -> str:
    """One-time task: starts at window start (also if the laptop was off/asleep, but only until the
    window ends), wakes the laptop, no run-time limit shorter than the window + grace, deletes itself."""
    w = get_window(cfg)
    if w is None:
        raise WindowError("no run_window configured in config/locked.yaml")
    user = user or _current_user()
    pythonw = paths.VENV / "Scripts" / "pythonw.exe"
    args = " ".join(f'"{a}"' if (" " in a or not a) else a for a in supervise_args(url))
    hours = int((w["grace_end"] - w["start"]).total_seconds() // 3600) + 3
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>GTA 6 Stream Auto Clipper: one launch-day run ({_fmt(w['start'])} to {_fmt(w['end'])}), then never again</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <StartBoundary>{_fmt(w['start'])}</StartBoundary>
      <EndBoundary>{_fmt(w['end'])}</EndBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
    <LogonTrigger>
      <StartBoundary>{_fmt(w['start'])}</StartBoundary>
      <EndBoundary>{_fmt(w['end'])}</EndBoundary>
      <Enabled>true</Enabled>
      <UserId>{escape(user)}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{escape(user)}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>true</WakeToRun>
    <ExecutionTimeLimit>PT{hours}H</ExecutionTimeLimit>
    <DeleteExpiredTaskAfter>PT6H</DeleteExpiredTaskAfter>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT5M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(str(pythonw))}</Command>
      <Arguments>{escape(args)}</Arguments>
      <WorkingDirectory>{escape(str(paths.ROOT))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _run(cmd: Sequence[str]) -> tuple[int, str, str]:
    import subprocess
    p = subprocess.run(list(cmd), capture_output=True, timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    dec = lambda b: (b or b"").decode("utf-8", "replace")  # noqa: E731
    return p.returncode, dec(p.stdout), dec(p.stderr)


def install(cfg: Any, url: str, runner=None) -> dict:
    run = runner or _run
    pythonw = paths.VENV / "Scripts" / "pythonw.exe"
    if not pythonw.exists():
        raise WindowError(f"{pythonw} not found (run scripts/setup.ps1 first)")
    if is_consumed():
        raise WindowError("the window was already used; run `python -m clipper window rearm` first")
    xf = paths.TMP / f"{TASK_NAME}.task.xml"
    xf.parent.mkdir(parents=True, exist_ok=True)
    xf.write_text(task_xml(cfg, url), encoding="utf-16")
    try:
        rc, out, err = run(["schtasks", "/Create", "/F", "/TN", TASK_NAME, "/XML", str(xf)])
    finally:
        xf.unlink(missing_ok=True)
    if rc != 0:
        hint = " Access was denied: run this once from an elevated PowerShell." if "denied" in (err + out).lower() else ""
        raise WindowError(f"schtasks /Create failed: rc={rc} {(err or out).strip()}{hint}")
    return {"output": out.strip()}


def remove(runner=None) -> bool:
    rc, out, err = (runner or _run)(["schtasks", "/Delete", "/F", "/TN", TASK_NAME])
    return rc == 0


def task_status(runner=None) -> dict:
    rc, out, err = (runner or _run)(["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"])
    if rc != 0:
        return {"registered": False}
    info: dict = {"registered": True}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info.setdefault(k.strip(), v.strip())
    return info


# ------------------------------------------------------------------ CLI
def main(argv: Sequence[str] | None = None) -> int:
    from .config import load_config
    ap = argparse.ArgumentParser(prog="clipper window")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install")
    i.add_argument("--url", default=None)
    sub.add_parser("remove")
    sub.add_parser("status")
    sub.add_parser("rearm")
    a = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config()
    try:
        if a.cmd == "install":
            url = a.url or (cfg.get("stream") or {}).get("url") or ""
            if not url:
                print("error: no stream URL: pass --url or set stream.url in config/settings.yaml")
                return 1
            print("installed:", install(cfg, url)["output"] or TASK_NAME)
            print("It starts once at the window start, wakes the laptop if needed, and stops after the window.")
            print("Keep the laptop plugged in and signed in. To cancel: python -m clipper window remove")
        elif a.cmd == "remove":
            print("removed" if remove() else "not registered")
        elif a.cmd == "rearm":
            rearm()
            print("re-armed: the window may run again (if its time has not passed)")
        else:
            w = get_window(cfg)
            if w is None:
                print("no run_window configured")
            else:
                print(f"window: {_fmt(w['start'])} -> {_fmt(w['end'])} (+{w['drain_grace_min']} min grace)")
                print(f"state: {status(cfg)}, consumed: {is_consumed()}, uploads in window: {w['uploads']}")
                ts = task_status()
                print("task:", ts.get("Status", "registered") if ts["registered"] else "not registered",
                      "| next run:", ts.get("Next Run Time", "-"))
    except WindowError as e:
        print(f"error: {e}")
        return 1
    return 0
