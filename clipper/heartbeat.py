"""Heartbeat files: each process beats from its MAIN LOOP (not a background thread) so a hung
loop goes silent. Long waits (ffmpeg, claude) poll with a timeout and call `hb.tick()`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import paths


def _path(name: str) -> Path:
    return paths.DATA.heartbeat / f"{name}.json"


def beat(name: str, **info) -> None:
    p = _path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"ts": time.time(), "pid": os.getpid(), **info}), encoding="utf-8")
    for _ in range(5):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:  # reader has it open on Windows
            time.sleep(0.05)
    tmp.unlink(missing_ok=True)


def read(name: str) -> dict | None:
    try:
        return json.loads(_path(name).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def age(name: str) -> float | None:
    """Seconds since the last beat, or None if never beaten."""
    d = read(name)
    return None if d is None else time.time() - float(d.get("ts", 0))


def clear(name: str) -> None:
    _path(name).unlink(missing_ok=True)


class Heartbeat:
    """Rate-limited beater. Call tick() often; it writes at most every `interval` seconds."""

    def __init__(self, name: str, interval: float = 30.0):
        self.name = name
        self.interval = interval
        self._last = 0.0
        self.info: dict = {}

    def tick(self, **info) -> None:
        if info:
            self.info.update(info)
        now = time.time()
        if now - self._last >= self.interval:
            beat(self.name, **self.info)
            self._last = now

    def force(self, **info) -> None:
        self._last = 0.0
        self.tick(**info)

    __call__ = tick
