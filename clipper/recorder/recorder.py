"""Live recorder: yt-dlp resolves the HLS playlist(s), ffmpeg copies them into fixed-length TS chunks.

Copy only (no decoding). Each closed segment (from ffmpeg's CSV segment list) is registered in
StateDB with cumulative stream seconds. On a stall (newest chunk not growing) or an ffmpeg exit the
partial chunk is registered, the URL is re-resolved (links expire) and recording continues with
the next index. When the stream is confirmed over, kv `recorder.done` is set and run() returns 0.

    python -m clipper.recorder.recorder --url URL [--chunk-seconds 300] [--max-chunks N]
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .. import crash, ffmpeg, paths
from .. import log as elog
from ..config import load_config
from ..heartbeat import Heartbeat
from ..models import Chunk
from ..state import StateDB
from . import stream

log = logging.getLogger(__name__)

CHUNK_RE = re.compile(r"chunk_(\d+)\.ts$", re.IGNORECASE)
LIVE = "is_live"
FFMPEG_LOG_MAX = 10_000_000


def _opt(section: Any, key: str, default: Any) -> Any:
    try:
        v = section.get(key, default)
    except AttributeError:
        return default
    return default if v is None else v


def chunk_name(idx: int) -> str:
    return f"chunk_{idx:05d}.ts"


class Recorder:
    def __init__(self, cfg: Any, state: StateDB, url: str, *, runner: stream.Runner | None = None,
                 chunk_seconds: float | None = None, max_chunks: int | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.state = state
        self.url = url
        self.runner = runner
        self.sleep = sleep
        rc = cfg.recorder
        self.chunk_seconds = float(chunk_seconds or rc.chunk_seconds)
        self.max_chunks = max_chunks
        self.stall_s = float(rc.stall_seconds)
        self.backoff = [float(x) for x in (_opt(rc, "reconnect_backoff_s", None) or [5])]
        self.end_checks = max(1, int(_opt(rc, "end_confirm_checks", 3)))
        self.end_check_interval = float(_opt(rc, "end_check_interval_s", 20))
        self.poll_s = float(_opt(rc, "poll_s", 1.0))
        self.min_chunk_s = float(_opt(rc, "min_chunk_s", 1.0))
        self.input_args = [str(a) for a in (_opt(rc, "input_args", None) or [])]
        self.loglevel = str(_opt(rc, "ffmpeg_loglevel", "warning"))
        self.hb = Heartbeat("recorder", interval=float(cfg.supervisor.heartbeat_s))

        self.proc: subprocess.Popen | None = None
        self.video_id: str | None = None
        self.title: str | None = None
        self.stream_id: str | None = None
        self.dir: Path | None = None
        self.next_idx = 0
        self.next_start = 0.0
        self.closed_this_run = 0     # closed (CSV) segments registered by this process
        self.sessions = 0

    # ------------------------------------------------------------------ public
    def run(self) -> int:
        self.hb.force(phase="waiting", url=self.url)
        # A previous stream's "done" flag must not make the worker exit (and the supervisor restart
        # it in a loop) while this run is still waiting for the next stream to go live.
        self.state.kv_set("recorder.done", False)
        info = stream.wait_until_live(self.url, self.cfg, self.hb, runner=self.runner, sleep=self.sleep)
        self._open_stream(info)
        attempt = 0
        last_reason = None
        while True:
            self.hb.tick(phase="resolving")
            p = stream.probe(self.url, self.cfg, want_urls=True, runner=self.runner, on_tick=self.hb.tick)
            if self._ours_live(p) and p["urls"]:
                if self.sessions:
                    delay = self.backoff[min(attempt, len(self.backoff) - 1)]
                    attempt += 1
                    log.warning("reconnecting after %s (attempt %d, wait %.1fs), next chunk %d",
                                last_reason, attempt, delay, self.next_idx)
                    elog.event("reconnect", stream_id=self.stream_id, reason=last_reason, attempt=attempt,
                               next_idx=self.next_idx, delay_s=delay)
                    stream.sleep_ticking(delay, self.hb, self.sleep)
                before = self.closed_this_run
                last_reason = self._session(p["urls"][:2])
                if last_reason == "max":
                    log.info("max chunks (%s) reached, stopping", self.max_chunks)
                    return 0
                if self.closed_this_run > before:
                    attempt = 0
                continue
            if p["status"] == "error":
                delay = self.backoff[min(attempt, len(self.backoff) - 1)]
                attempt += 1
                log.warning("yt-dlp probe failed (wait %.1fs): %s", delay, p.get("error", "")[-300:])
                stream.sleep_ticking(delay, self.hb, self.sleep)
                continue
            # This probe already says "not live (for our video)"; confirm with more checks.
            if self._confirm_end(first=p):
                self._finish()
                return 0

    # ------------------------------------------------------------------ stream / resume
    def _ours_live(self, st: dict) -> bool:
        return st.get("status") == LIVE and st.get("video_id") == self.video_id

    def _open_stream(self, info: dict) -> None:
        self.video_id = info["video_id"]
        self.title = info.get("title")
        cur = self.state.current_stream()
        if cur and cur.startswith(f"{self.video_id}-"):
            self.stream_id = cur
        else:
            self.stream_id = f"{self.video_id}-{time.strftime('%Y%m%d')}"
        self.state.set_current_stream(self.stream_id)
        self.state.kv_set("recorder.done", False)
        self.state.kv_set("recorder.title", self.title)
        self._kill_orphan_ffmpeg()
        last = self.state.last_chunk_idx(self.stream_id)
        if last >= 0:
            c = self.state.get_chunk(self.stream_id, last)
            self.next_idx = last + 1
            self.next_start = c.end_s if c else 0.0
        self.dir = paths.DATA.chunks / self.stream_id
        self.dir.mkdir(parents=True, exist_ok=True)
        recovered = self._register_leftovers()
        log.info("recording %s (%s) from chunk %d at %.1fs%s", self.stream_id, self.title, self.next_idx,
                 self.next_start, f", recovered {recovered} chunk(s)" if recovered else "")
        self.hb.force(phase="recording", stream_id=self.stream_id, chunk=self.next_idx)

    def _kill_orphan_ffmpeg(self) -> None:
        """An ffmpeg left running by a crashed recorder would keep writing into our files."""
        pid = self.state.kv_get("recorder.ffmpeg_pid")
        if not pid:
            return
        try:
            import psutil
            p = psutil.Process(int(pid))
            if p.name().lower().startswith("ffmpeg"):
                log.warning("killing orphan ffmpeg pid %s", pid)
                ffmpeg.kill_tree(int(pid))
                p.wait(timeout=10)
        except Exception:  # gone already / access denied
            pass
        self.state.kv_set("recorder.ffmpeg_pid", None)

    def _confirm_end(self, first: dict) -> bool:
        count = 1
        log.info("stream not live (status=%s, id=%s); confirming", first.get("status"), first.get("video_id"))
        while count < self.end_checks:
            stream.sleep_ticking(self.end_check_interval, self.hb, self.sleep)
            st = stream.live_status(self.url, self.cfg, runner=self.runner, on_tick=self.hb.tick)
            if self._ours_live(st):
                log.info("stream live again")
                return False
            if st["status"] == "error":
                continue
            count += 1
        return True

    def _finish(self) -> None:
        chunks = self.state.chunks(self.stream_id)
        self.state.kv_set("recorder.done", True)
        elog.event("stream_end", stream_id=self.stream_id, chunks=len(chunks), total_s=round(self.next_start, 3),
                   last_idx=self.next_idx - 1)
        log.info("stream ended: %s, %d chunks, %.1fs", self.stream_id, len(chunks), self.next_start)
        self.hb.force(phase="done")

    # ------------------------------------------------------------------ one ffmpeg session
    def _ffmpeg_cmd(self, urls: list[str], start_idx: int, csv_path: Path) -> list[str]:
        cmd = [ffmpeg.FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", self.loglevel]
        for u in urls:
            cmd += [*self.input_args, "-i", u]
        if len(urls) >= 2:
            cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        else:
            cmd += ["-map", "0:v:0?", "-map", "0:a:0?"]
        cmd += ["-c", "copy", "-f", "segment", "-segment_time", f"{self.chunk_seconds:g}",
                "-segment_format", "mpegts", "-segment_format_options", "flush_packets=1",
                "-reset_timestamps", "1",
                "-segment_start_number", str(start_idx),
                "-segment_list", str(csv_path), "-segment_list_type", "csv",
                str(self.dir / "chunk_%05d.ts")]
        return cmd

    def _open_ffmpeg_log(self):
        p = paths.DATA.logs / "recorder_ffmpeg.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            if p.stat().st_size > FFMPEG_LOG_MAX:
                p.unlink()
        except OSError:
            pass
        f = open(p, "ab")
        f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {self.stream_id} from chunk {self.next_idx}\n"
                .encode())
        f.flush()
        return f

    def _session(self, urls: list[str]) -> str:
        """Record until stall / ffmpeg exit / max chunks. Returns 'stall' | 'exit' | 'eof' | 'max'."""
        self.sessions += 1
        start_idx = self.next_idx
        self.dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.dir / f"segments_{start_idx:05d}.csv"
        csv_path.unlink(missing_ok=True)
        cmd = self._ffmpeg_cmd(urls, start_idx, csv_path)
        flog = self._open_ffmpeg_log()
        log.info("ffmpeg session %d: %d input(s), chunk %d, %gs segments", self.sessions, len(urls),
                 start_idx, self.chunk_seconds)
        self.proc = ffmpeg.popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=flog)
        self.state.kv_set("recorder.ffmpeg_pid", self.proc.pid)
        seen: set[str] = set()
        last_size = -1
        last_growth = time.monotonic()
        reason = "exit"
        try:
            while True:
                self.hb.tick(phase="recording", stream_id=self.stream_id, chunk=self.next_idx)
                self._consume_csv(csv_path, seen)
                if self.max_chunks is not None and self.closed_this_run >= self.max_chunks:
                    reason = "max"
                    break
                rc = self.proc.poll()
                if rc is not None:
                    reason = "eof" if rc == 0 else "exit"
                    log.warning("ffmpeg exited with %s", rc)
                    break
                cur = self.dir / chunk_name(self.next_idx)
                try:
                    size = cur.stat().st_size
                except OSError:
                    size = -1
                now = time.monotonic()
                if size != last_size:
                    last_size, last_growth = size, now
                elif now - last_growth > self.stall_s:
                    reason = "stall"
                    log.warning("stall: %s not growing for %.0fs", cur.name, now - last_growth)
                    break
                self.sleep(self.poll_s)
        finally:
            self._stop_ffmpeg()
            flog.close()
            self.state.kv_set("recorder.ffmpeg_pid", None)
        if reason == "max":
            self._discard_from(self.next_idx)
        else:
            self._consume_csv(csv_path, seen)
            self._register_leftovers()
        return reason

    def _stop_ffmpeg(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        if p.poll() is None:
            ffmpeg.kill_tree(p)
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            log.error("ffmpeg pid %s did not exit after kill", p.pid)

    # ------------------------------------------------------------------ chunk bookkeeping
    def _consume_csv(self, csv_path: Path, seen: set[str]) -> None:
        """Register every newly closed segment listed in ffmpeg's CSV (complete lines only)."""
        try:
            text = csv_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        if not text.endswith("\n"):
            text = text[: text.rfind("\n") + 1]
        for row in csv.reader(text.splitlines()):
            if not row:
                continue
            name = Path(row[0].strip()).name
            if name in seen:
                continue
            seen.add(name)
            m = CHUNK_RE.search(name)
            if not m:
                continue
            idx = int(m.group(1))
            if idx < self.next_idx:
                continue
            if self._register(idx, self.dir / name, min_s=0.0):
                self.closed_this_run += 1
            else:  # empty closed segment: drop it, keep numbering moving so stall tracking follows
                (self.dir / name).unlink(missing_ok=True)
                self.next_idx = idx + 1

    def _chunk_files(self, min_idx: int) -> list[tuple[int, Path]]:
        out = []
        for p in self.dir.glob("chunk_*.ts"):
            m = CHUNK_RE.search(p.name)
            if m and int(m.group(1)) >= min_idx:
                out.append((int(m.group(1)), p))
        return sorted(out)

    def _register_leftovers(self) -> int:
        """Register unregistered chunk files (the partial after a stall/exit, or files from a crash)."""
        n = 0
        for idx, p in self._chunk_files(self.next_idx):
            if self._register(idx, p, min_s=self.min_chunk_s):
                n += 1
            else:
                p.unlink(missing_ok=True)
        return n

    def _discard_from(self, idx: int) -> None:
        for _, p in self._chunk_files(idx):
            p.unlink(missing_ok=True)

    def _register(self, idx: int, path: Path, min_s: float) -> bool:
        try:
            dur = ffmpeg.duration(path)
        except Exception as e:
            log.warning("ffprobe failed on %s: %s", path.name, e)
            dur = 0.0
        if dur <= 0 or dur < min_s:
            log.info("skipping %s (%.2fs of media)", path.name, dur)
            return False
        c = Chunk(idx=idx, path=str(path), start_s=round(self.next_start, 3), duration_s=round(dur, 3),
                  recorded_at=time.time(), stream_id=self.stream_id)
        self.state.add_chunk(c)
        self.next_start = c.start_s + c.duration_s
        self.next_idx = idx + 1
        elog.event("chunk_recorded", stream_id=self.stream_id, idx=idx, path=str(path), start_s=c.start_s,
                   duration_s=c.duration_s)
        log.info("chunk %d registered: start %.1fs, %.1fs", idx, c.start_s, c.duration_s)
        self.hb.tick(chunk=self.next_idx)
        return True


# ---------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clipper record", description="Record a YouTube live stream into chunks.")
    ap.add_argument("--url", help="stream or channel /live URL (default: stream.url in settings.yaml)")
    ap.add_argument("--chunk-seconds", type=float, default=None, help="segment length (testing)")
    ap.add_argument("--max-chunks", type=int, default=None, help="stop after N closed chunks (testing)")
    args = ap.parse_args(argv)
    elog.setup_logging("recorder")
    crash.install_crash_handler("recorder")
    cfg = load_config()
    url = args.url or cfg.stream.get("url") or ""
    if not url:
        log.error("no stream URL: pass --url or set stream.url in config/settings.yaml")
        return 2
    paths.DATA.ensure()
    state = StateDB()
    try:
        return Recorder(cfg, state, url, chunk_seconds=args.chunk_seconds, max_chunks=args.max_chunks).run()
    finally:
        state.close()


if __name__ == "__main__":
    sys.exit(main())
