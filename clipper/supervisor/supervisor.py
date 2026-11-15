"""Supervisor: runs the recorder and the worker inside one 2 GB job object and keeps them alive.

    python -m clipper supervise [--url URL] [--no-repair] [--once]

- Restarts a process that exits non-zero, with growing backoff (reset after a stable run).
- Hang detection: heartbeat older than `hang_s` (after a startup grace) => kill tree + restart.
- The recorder exiting 0 means the stream ended: it isn't restarted. The worker drains the
  backlog and exits 0 once `recorder.done` is set and nothing is left; then the supervisor stops.
- The same worker crash signature `crash_count` times within `crash_window_s` => repair agent
  (in a background thread, so the recorder keeps being watched). After the per-stream repair
  budget is used up the worker only restarts slowly and publishing stays paused.
- The supervisor itself runs outside the job and keeps the laptop awake while it runs.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

if __package__ in (None, ""):  # started as a script (scheduled-task fallback): make `clipper` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clipper import crash, heartbeat, log as elog, paths  # noqa: E402
from clipper.supervisor import nosleep  # noqa: E402
from clipper.supervisor.jobobject import JobObject, spawn_in_job  # noqa: E402
from clipper.supervisor.procs import (BACKOFF, CAPPED, DONE, REPAIRING, RUNNING, STOPPED,  # noqa: E402
                                      ManagedProc, log_tail)
from clipper.supervisor.repair import release_repair_hold  # noqa: E402

log = logging.getLogger("clipper.supervisor")

RECORDER, WORKER = "recorder", "worker"


def default_commands(url: str) -> dict[str, list[str]]:
    py = str(paths.VENV_PY)
    return {RECORDER: [py, "-m", "clipper", "record", "--url", url],
            WORKER: [py, "-m", "clipper", "work"]}


class Supervisor:
    def __init__(self, cfg, url: str, state, spawn: Callable | None = None, clock: Callable[[], float] | None = None,
                 *, commands: dict[str, Sequence[str]] | None = None, repair_agent=None, repair: bool = True,
                 once: bool = False, job: JobObject | None = None):
        self.cfg = cfg
        self.url = url
        self.state = state
        self.clock = clock or time.monotonic
        self.once = once
        self.repair_enabled = repair and not once
        self.agent = repair_agent
        self.job = job
        self._own_job = job is None
        self._spawn = spawn
        s = cfg.supervisor
        self.backoff = [float(x) for x in s.get("backoff_s", [5, 15, 30, 60, 120, 300])]
        self.hang_s = float(s.get("hang_s", 120))
        self.grace_s = float(s.get("startup_grace_s", self.hang_s))
        self.stable_reset_s = float(s.get("stable_reset_s", 600))
        self.crash_window_s = float(s.get("crash_window_s", 600))
        self.crash_count = int(s.get("crash_count", 3))
        self.poll_s = float(s.get("poll_s", 1.0))
        self.mem_log_s = float(s.get("mem_log_s", 300))
        self.slow_restart_s = float(s.get("slow_restart_s", 900))
        self.tail_lines = int(s.get("log_tail_lines", 200))
        cmds = commands or default_commands(url)
        self.procs: dict[str, ManagedProc] = {
            name: ManagedProc(name, cmds[name], self._do_spawn, self.clock) for name in (RECORDER, WORKER)}
        self._stop = threading.Event()
        self._stop_reason = ""
        self._repair_thread: threading.Thread | None = None
        self._repair_result: tuple[bool, str] | None = None
        self._last_mem = float("-inf")

    # -------------------------------------------------------------- plumbing
    def _do_spawn(self, cmd, **kw):
        if self._spawn is not None:
            return self._spawn(cmd, **kw)
        return spawn_in_job(cmd, self.job, **kw)

    def request_stop(self, reason: str = "requested") -> None:
        self._stop_reason = self._stop_reason or reason
        self._stop.set()

    @property
    def recorder(self) -> ManagedProc:
        return self.procs[RECORDER]

    @property
    def worker(self) -> ManagedProc:
        return self.procs[WORKER]

    # -------------------------------------------------------------- main loop
    def run(self) -> int:
        if self.job is None:
            self.job = JobObject(self.cfg.locked.memory_limit_mb)
        if self.repair_enabled and self.agent is None:
            from clipper.supervisor.repair import RepairAgent
            self.agent = RepairAgent(self.cfg, self.state, job=self.job)
        nosleep.keep_awake()
        elog.event("supervisor_start", url=self.url, memory_limit_mb=self.cfg.locked.memory_limit_mb,
                   repair=self.repair_enabled, once=self.once)
        log.info("supervisor started (url=%s, job limit %s MB)", self.url, self.cfg.locked.memory_limit_mb)
        try:
            for p in self.procs.values():
                self._start(p)
            while not self._stop.is_set():
                try:
                    self._tick()
                except Exception:  # the supervisor must survive its own hiccups (db busy, etc.)
                    log.exception("supervisor tick failed")
                if self._finished():
                    self.request_stop("finished")
                    break
                self._stop.wait(self.poll_s)
        finally:
            self._shutdown()
        return 0

    def _tick(self) -> None:
        self._check_repair_done()
        for p in self.procs.values():
            if p.status == RUNNING:
                rc = p.poll()
                if rc is None:
                    self._check_hang(p)
                else:
                    self._on_exit(p, rc)
            elif p.status in (BACKOFF, CAPPED) and self.clock() >= p.next_start:
                self._start(p)
        self._log_memory()

    def _finished(self) -> bool:
        rec, wrk = self.recorder, self.worker
        if self.once:
            return all(p.status in (DONE, STOPPED) for p in self.procs.values())
        if rec.status != DONE:
            return False
        if wrk.status in (DONE, STOPPED):
            return True
        return wrk.capped and wrk.status == CAPPED  # stream over, worker can't be repaired: stop

    def _start(self, p: ManagedProc) -> None:
        try:
            p.start()
        except Exception as e:
            log.exception("failed to start %s", p.name)
            delay = p.failure_delay(self.backoff, 0.0, self.stable_reset_s)
            p.schedule_restart(delay)
            elog.event("spawn_failed", name=p.name, error=str(e)[:500], retry_in_s=delay)
            return
        log.info("started %s pid=%s (start #%d)", p.name, p.pid, p.starts)
        elog.event("proc_start", name=p.name, pid=p.pid, starts=p.starts)

    # -------------------------------------------------------------- exits
    def _unprocessed(self) -> bool:
        try:
            sid = self.state.current_stream()
            return bool(sid) and self.state.next_chunk_to_process(sid) is not None
        except Exception:
            log.exception("state check failed")
            return True

    def _fresh_report(self, p: ManagedProc) -> dict | None:
        rep = crash.read_report(p.name)
        if rep and float(rep.get("ts", 0)) >= p.started_wall - 1.0:
            return rep
        return None

    def _on_exit(self, p: ManagedProc, rc: int) -> None:
        uptime = self.clock() - p.started_at
        elog.event("proc_exit", name=p.name, pid=p.pid, rc=rc, uptime_s=round(uptime, 1))
        if rc == 0:
            p.streak = 0
            if p.name == RECORDER:
                p.status = DONE
                log.info("recorder exited 0: stream ended, letting the worker drain")
                elog.event("stream_end", source="supervisor")
                return
            if self.once or (self.recorder.status == DONE and not self._unprocessed()):
                p.status = DONE
                log.info("worker finished (backlog drained)")
                elog.event("worker_drained")
                return
            delay = self.backoff[0] if self.backoff else 0.0
            p.schedule_restart(delay)
            elog.event("restart_scheduled", name=p.name, delay_s=delay, reason="clean_exit")
            return
        rep = self._fresh_report(p)
        if rep is None:
            rep = {"signature": f"exit:{rc}", "type": "Exit", "message": f"{p.name} exited with code {rc}",
                   "traceback": ""}
        self._on_failure(p, rc, rep, uptime)

    def _on_failure(self, p: ManagedProc, rc: int | None, report: dict, uptime: float) -> None:
        sig = str(report.get("signature") or f"exit:{rc}")
        log.warning("%s failed (rc=%s, %s) after %.0fs", p.name, rc, sig, uptime)
        elog.event("crash", name=p.name, rc=rc, signature=sig, message=str(report.get("message", ""))[:500],
                   uptime_s=round(uptime, 1))
        if self.once:
            p.status = DONE
            return
        delay = p.failure_delay(self.backoff, uptime, self.stable_reset_s)
        count = p.record_crash(sig, self.crash_window_s)
        if p.name == WORKER:
            if p.capped:
                self._enter_capped(p, sig)
                return
            if self.repair_enabled and self.agent is not None and count >= self.crash_count:
                p.forget_crashes(sig)
                if self.agent.exhausted():
                    self.agent.attempt(report, "")  # logs the cap + attention, keeps the hold
                    self._enter_capped(p, sig)
                else:
                    self._start_repair(p, report)
                return
        p.schedule_restart(delay)
        elog.event("restart_scheduled", name=p.name, delay_s=delay, signature=sig, streak=p.streak)

    def _enter_capped(self, p: ManagedProc, sig: str) -> None:
        p.capped = True
        if self.slow_restart_s > 0:
            p.status = CAPPED
            p.next_start = self.clock() + self.slow_restart_s
        else:
            p.status = STOPPED
        log.error("worker repair budget used up (%s); slow restart in %.0fs", sig, self.slow_restart_s)
        elog.event("worker_capped", signature=sig, slow_restart_s=self.slow_restart_s)

    # -------------------------------------------------------------- hangs
    def _check_hang(self, p: ManagedProc) -> None:
        up = p.uptime()
        if up < self.grace_s:
            return
        age = heartbeat.age(p.name)
        if age is not None:
            p.last_beat_wall = time.time() - age
        # a momentarily unreadable beat file (replace race) falls back to the last beat seen
        eff = time.time() - p.last_beat_wall if p.last_beat_wall else up
        if age is None and not p.last_beat_wall:
            eff = up
        if eff <= self.hang_s:
            return
        info = heartbeat.read(p.name) or {}
        stage = info.get("stage") or info.get("step") or "unknown"
        log.error("%s hung: no heartbeat for %.0fs (pid %s); killing", p.name, eff, p.pid)
        elog.event("hang", name=p.name, pid=p.pid, heartbeat_age_s=None if age is None else round(age, 1),
                   uptime_s=round(up, 1), stage=stage)
        rc = p.kill()
        report = {"signature": f"hang@{stage}", "type": "Hang",
                  "message": f"{p.name} sent no heartbeat for {eff:.0f}s (last beat: {info})",
                  "traceback": "", "heartbeat": info}
        self._on_failure(p, rc, report, up)

    # -------------------------------------------------------------- repair
    def _start_repair(self, p: ManagedProc, report: dict) -> None:
        p.status = REPAIRING
        sig = str(report.get("signature"))
        tail = log_tail(p.name, self.tail_lines)
        log.warning("worker crashed %d times with %s: starting the repair agent", self.crash_count, sig)
        elog.event("repair_start", signature=sig)
        self._repair_result = None

        def target():
            try:
                ok = bool(self.agent.attempt(report, tail))
            except Exception:
                log.exception("repair agent raised")
                ok = False
            self._repair_result = (ok, sig)

        self._repair_thread = threading.Thread(target=target, name="repair", daemon=True)
        self._repair_thread.start()

    def _check_repair_done(self) -> None:
        t = self._repair_thread
        if t is None or t.is_alive():
            return
        self._repair_thread = None
        ok, sig = self._repair_result or (False, "?")
        w = self.worker
        if ok:
            release_repair_hold(self.state)
            w.streak = 0
            w.forget_crashes()
            elog.event("repair_applied", signature=sig)
            log.info("repair for %s committed: restarting the worker to process the backlog", sig)
            self._start(w)
        elif self.agent is not None and self.agent.exhausted():
            self._enter_capped(w, sig)
        else:
            delay = w.failure_delay(self.backoff, 0.0, self.stable_reset_s)
            w.schedule_restart(delay)
            elog.event("restart_scheduled", name=w.name, delay_s=delay, signature=sig, reason="repair_failed")

    # -------------------------------------------------------------- memory / shutdown
    def _log_memory(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._last_mem < self.mem_log_s:
            return
        self._last_mem = now
        if self.job is None or not self.job.handle:
            return
        try:
            peak = self.job.peak_memory()
            cur = self.job.current_memory()
        except Exception:
            log.exception("job memory query failed")
            return
        elog.event("job_memory", peak_mb=round(peak / 2**20, 1), current_mb=round(cur / 2**20, 1),
                   limit_mb=self.cfg.locked.memory_limit_mb)

    def _shutdown(self) -> None:
        for p in self.procs.values():
            if p.status == RUNNING and p.poll() is None:
                log.info("stopping %s (pid %s)", p.name, p.pid)
                p.kill()
            if p.status in (RUNNING, BACKOFF, CAPPED, REPAIRING):
                p.status = STOPPED
        if self._repair_thread is not None and self._repair_thread.is_alive():
            log.warning("waiting for the repair attempt to finish")
            self._repair_thread.join(timeout=30)
        self._log_memory(force=True)
        if self._own_job and self.job is not None:
            self.job.close()
        nosleep.allow_sleep()
        elog.event("supervisor_stop", reason=self._stop_reason or "exit")
        log.info("supervisor stopped (%s)", self._stop_reason or "exit")


# ---------------------------------------------------------------- entry point
def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="clipper supervise")
    ap.add_argument("--url", default=None, help="stream URL (default: stream.url in settings.yaml)")
    ap.add_argument("--no-repair", action="store_true", help="never launch the Claude repair agent")
    ap.add_argument("--once", action="store_true", help="run recorder+worker once, no restarts (testing)")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.stderr is None or sys.stdout is None:  # pythonw (scheduled task): no console
        paths.DATA.logs.mkdir(parents=True, exist_ok=True)
        f = open(paths.DATA.logs / "supervisor.console.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f
    from clipper.env import apply_to_current_process
    apply_to_current_process()
    elog.setup_logging("supervisor")
    crash.install_crash_handler("supervisor")
    from clipper.config import load_config
    from clipper.state import StateDB
    cfg = load_config()
    url = args.url or (cfg.get("stream") or {}).get("url") or ""
    if not url:
        elog.attention("Supervisor: no stream URL (pass --url or set stream.url in config/settings.yaml)")
        return 2
    state = StateDB()
    sup = Supervisor(cfg, url, state, repair=not args.no_repair, once=args.once)
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, lambda *_a, _n=name: sup.request_stop(_n))
            except (ValueError, OSError):
                pass
    try:
        return sup.run()
    finally:
        state.close()


if __name__ == "__main__":
    raise SystemExit(main())
