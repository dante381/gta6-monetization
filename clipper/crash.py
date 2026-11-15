"""Crash reports read by the supervisor to detect repeated identical crashes.

Each process calls install_crash_handler("<name>") at startup. An unhandled exception writes
data/logs/last_error_<name>.json:
    {"name", "ts", "type", "message", "signature", "location", "traceback"}
signature = "<ExceptionType>@<file>:<function>" of the innermost project frame, so the same
bug gives the same signature even if line numbers or messages vary slightly.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from . import paths


def error_path(name: str) -> Path:
    return paths.DATA.logs / f"last_error_{name}.json"


def signature_of(exc: BaseException) -> tuple[str, str]:
    tb = traceback.extract_tb(exc.__traceback__)
    loc = ""
    root = str(paths.ROOT).lower()
    for fr in reversed(tb):
        if fr.filename.lower().startswith(root) and ".venv" not in fr.filename.lower():
            rel = Path(fr.filename).resolve().relative_to(paths.ROOT).as_posix()
            loc = f"{rel}:{fr.name}"
            break
    if not loc and tb:
        loc = f"{Path(tb[-1].filename).name}:{tb[-1].name}"
    return f"{type(exc).__name__}@{loc}", loc


def write_report(name: str, exc: BaseException) -> dict:
    sig, loc = signature_of(exc)
    rec = {"name": name, "ts": time.time(), "type": type(exc).__name__, "message": str(exc)[:2000],
           "signature": sig, "location": loc,
           "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:]}
    p = error_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def read_report(name: str) -> dict | None:
    try:
        return json.loads(error_path(name).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def install_crash_handler(name: str) -> None:
    prev = sys.excepthook

    def hook(etype, value, tb):
        if not issubclass(etype, KeyboardInterrupt):
            try:
                write_report(name, value)
            except Exception:  # never mask the original crash
                pass
        prev(etype, value, tb)

    sys.excepthook = hook
