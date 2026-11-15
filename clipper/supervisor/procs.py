"""One supervised child process (recorder or worker): spawn, poll, kill, restart bookkeeping."""
from __future__ import annotations

import logging
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Callable, Sequence

from clipper import heartbeat, paths
from clipper.env import subprocess_env
from clipper.ffmpeg import kill_tree

log = logging.getLogger(__name__)

SpawnFn = Callable[..., subprocess.Popen]

# status values
IDLE, RUNNING, BACKOFF, DONE, REPAIRING, CAPPED, STOPPED = (
    "idle", "running", "backoff", "done", "repairing", "capped", "stopped")


def console_log_path(name: str) -> Path:
    return paths.DATA.logs / f"{name}.console.log"


class ManagedProc:
    def __init__(self, name: str, cmd: Sequence[str], spawn: SpawnFn, clock: Callable[[], float]):
        self.name = name
        self.cmd = [str(c) for c in cmd]
        self.spawn = spawn
        self.clock = clock
        self.popen: subprocess.Popen | None = None
        self.status = IDLE
        self.started_at = 0.0        # clock() at start
        self.started_wall = 0.0      # time.time() at start (crash report freshness)
        self.last_beat_wall = 0.0    # wall time of the last heartbeat seen this run (0 = none yet)
        self.next_start = 0.0        # clock() when a BACKOFF/CAPPED proc may start again
        self.streak = 0              # consecutive failures (backoff index)
        self.starts = 0
        self.capped = False          # repair budget exhausted for a repeating crash
        self.crash_times: dict[str, deque] = {}

    # -------------------------------------------------------------- lifecycle
    @property
    def pid(self) -> int | None:
        return self.popen.pid if self.popen else None

    def uptime(self) -> float:
        return self.clock() - self.started_at if self.status == RUNNING else 0.0

    def start(self) -> subprocess.Popen:
        heartbeat.clear(self.name)  # a stale beat from a previous run must not count
        logf = console_log_path(self.name)
        logf.parent.mkdir(parents=True, exist_ok=True)
        with open(logf, "ab") as out:
            out.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(self.cmd)}\n".encode())
            out.flush()
            self.popen = self.spawn(self.cmd, cwd=str(paths.ROOT), env=subprocess_env(),
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        self.status = RUNNING
        self.started_at = self.clock()
        self.started_wall = time.time()
        self.last_beat_wall = 0.0
        self.starts += 1
        return self.popen

    def poll(self) -> int | None:
        return None if self.popen is None else self.popen.poll()

    def kill(self, wait_s: float = 10.0) -> int | None:
        if self.popen is None:
            return None
        if self.popen.poll() is None:
            kill_tree(self.popen)
            try:
                self.popen.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                log.error("%s (pid %s) did not die after kill", self.name, self.popen.pid)
        return self.popen.poll()

    # -------------------------------------------------------------- restart policy
    def schedule_restart(self, delay: float) -> None:
        self.status = BACKOFF
        self.next_start = self.clock() + max(0.0, delay)

    def failure_delay(self, backoff: Sequence[float], uptime: float, stable_reset_s: float) -> float:
        """Backoff for the next start after a failure; a long stable run resets the streak."""
        if uptime >= stable_reset_s:
            self.streak = 0
        delay = float(backoff[min(self.streak, len(backoff) - 1)]) if backoff else 0.0
        self.streak += 1
        return delay

    def record_crash(self, signature: str, window_s: float) -> int:
        """Remember one crash; return how many crashes with this signature fall in the window."""
        now = self.clock()
        q = self.crash_times.setdefault(signature, deque())
        q.append(now)
        while q and now - q[0] > window_s:
            q.popleft()
        return len(q)

    def forget_crashes(self, signature: str | None = None) -> None:
        if signature is None:
            self.crash_times.clear()
        else:
            self.crash_times.pop(signature, None)


def log_tail(name: str, lines: int = 200, max_chars: int = 20000) -> str:
    """Tail of the process's text log plus its console (stdout/stderr) log."""
    parts = []
    for p in (paths.DATA.logs / f"{name}.log", console_log_path(name)):
        try:
            with open(p, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - max_chars * 2))
                text = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        tail = "\n".join(text.splitlines()[-lines:])
        parts.append(f"--- {p.name} (last {lines} lines) ---\n{tail}")
    out = "\n\n".join(parts)
    return out[-max_chars:] if out else "(no log output found)"
