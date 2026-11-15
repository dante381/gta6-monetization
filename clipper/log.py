"""Logging: per-process text log + one shared JSONL event log (clips, scores, uploads)."""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from typing import Any

from . import paths

_configured: set[str] = set()


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging to data/logs/<name>.log (rotating) + stderr. Idempotent."""
    if name not in _configured:
        paths.DATA.logs.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        fh = logging.handlers.RotatingFileHandler(paths.DATA.logs / f"{name}.log", maxBytes=5_000_000,
                                                  backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                   for h in root.handlers):
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            root.addHandler(sh)
        _configured.add(name)
    return logging.getLogger(name)


def event(kind: str, **fields: Any) -> None:
    """Append one structured event to data/logs/clips.jsonl (for tuning and the run report).

    kinds used: chunk_recorded, chunk_processed, candidate, clip_produced, upload, upload_skipped,
    would_upload, crash, repair, attention, reconnect, stream_end, claude_unavailable
    """
    paths.DATA.logs.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.time(), "kind": kind, **fields}
    line = json.dumps(rec, default=str)
    with open(paths.DATA.logs / "clips.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def attention(message: str, **fields: Any) -> None:
    """Something needs the human. Goes to the event log and data/logs/ATTENTION.txt."""
    event("attention", message=message, **fields)
    with open(paths.DATA.logs / "ATTENTION.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")
    logging.getLogger("attention").warning(message)
