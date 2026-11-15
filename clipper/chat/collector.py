"""Live-chat collector: reads the PUBLIC chat of the live stream and stores per-bucket aggregates.

    python -m clipper chat [--video-id ID] [--stream-id ID]

Privacy: author identifiers are only hashed in memory to count distinct chatters per bucket and are
discarded when the bucket closes; they never reach the database or a log. Message text is stored
only as a few short sanitized samples per bucket. Nothing about chat content is logged.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Protocol

from .. import crash, paths
from .. import log as elog
from ..config import load_config
from ..heartbeat import Heartbeat
from ..state import StateDB
from .scoring_text import clip_request, hype_score, sanitize

log = logging.getLogger(__name__)

# ChatSource: an iterator of (epoch_seconds, author_key, text); may block between messages.
ChatMsg = tuple[float, str, str]


class ChatSource(Protocol):
    def __iter__(self) -> Iterator[ChatMsg]: ...


SourceFactory = Callable[[str], Iterable[ChatMsg]]


def library_source(video_id: str) -> Iterator[ChatMsg]:
    """Adapter over pytchat (reads the public chat like the website; no API key, no sign-in).

    The author's channel id is only used as an opaque de-dup key by the collector (hashed, never stored).
    """
    import pytchat  # imported lazily: not needed by tests that inject a fake source
    chat = pytchat.create(video_id=video_id, interruptable=False)
    try:
        while chat.is_alive():
            for m in chat.get().sync_items():
                text = m.message
                if not text:
                    continue
                yield float(m.timestamp) / 1000.0, str(m.author.channelId or ""), str(text)
            time.sleep(1.0)
    finally:
        try:
            chat.terminate()
        except Exception:  # noqa: BLE001
            pass


def _opt(section: Any, key: str, default: Any) -> Any:
    try:
        v = section.get(key, default)
    except AttributeError:
        return default
    return default if v is None else v


class _Bucket:
    __slots__ = ("msgs", "hype", "clips", "samples", "n_samples", "authors")

    def __init__(self) -> None:
        self.msgs = 0           # unflushed deltas
        self.hype = 0.0
        self.clips = 0
        self.samples: list[str] = []
        self.n_samples = 0      # samples taken so far (flushed or not)
        self.authors: set[bytes] = set()

    def dirty(self) -> bool:
        return bool(self.msgs)


class Collector:
    def __init__(self, cfg: Any, state: StateDB, video_id: str, stream_id: str, *,
                 source_factory: SourceFactory | None = None, clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] | None = None, max_seconds: float | None = None,
                 backoff: list[float] | None = None):
        self.cfg = cfg
        self.ccfg = cfg.chat
        self.state = state
        self.video_id = video_id
        self.stream_id = stream_id
        self.factory = source_factory or library_source
        self.clock = clock or time.time
        self.max_seconds = max_seconds
        self.bucket_s = max(1, int(_opt(self.ccfg, "bucket_s", 5)))
        self.poll_s = float(_opt(self.ccfg, "collector_poll_s", 2.0))
        self.n_samples = int(_opt(self.ccfg, "sample_per_bucket", 3))
        self.backoff = backoff or [float(x) for x in _opt(self.ccfg, "reconnect_backoff_s", [2, 5, 15, 30, 60])]
        self.warn_after = 5
        self.salt = os.urandom(16)
        self.hb = Heartbeat("chat", interval=min(5.0, float(cfg.supervisor.get("heartbeat_s", 30))))
        self.buckets: dict[int, _Bucket] = {}
        self.total = 0
        self.failures = 0
        self.warned = False
        self.last_msg_at: float | None = None
        self._lag_sum = 0.0

    # ------------------------------------------------------------------ aggregation
    def add(self, ts: float, author: str, text: str) -> None:
        t = int(ts // self.bucket_s * self.bucket_s)
        b = self.buckets.get(t)
        if b is None:
            b = self.buckets[t] = _Bucket()
        b.msgs += 1
        self.total += 1
        b.authors.add(hashlib.blake2b(author.encode("utf-8", "ignore"), key=self.salt, digest_size=8).digest())
        b.hype += hype_score(text, self.ccfg)
        if clip_request(text, self.ccfg):
            b.clips += 1
        if b.n_samples < self.n_samples:
            s = sanitize(text, self.ccfg)
            if s:
                b.samples.append(s)
                b.n_samples += 1

    def flush(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        for t, b in sorted(self.buckets.items()):
            if b.dirty():
                self.state.chat_add(self.stream_id, t, b.msgs, len(b.authors), b.hype, b.clips, b.samples)
                b.msgs, b.hype, b.clips, b.samples = 0, 0.0, 0, []
        # buckets older than a minute can no longer receive realistic late messages: forget them
        for t in [t for t in self.buckets if t + self.bucket_s + 60 < now]:
            del self.buckets[t]

    # ------------------------------------------------------------------ source thread
    def _reader(self, q: "queue.Queue", stop: threading.Event) -> None:
        try:
            for msg in self.factory(self.video_id):
                if stop.is_set():
                    return
                q.put(("msg", msg))
            q.put(("end", None))
        except BaseException as e:  # noqa: BLE001 - reported by type name only (never message content)
            q.put(("err", type(e).__name__))

    def _done(self) -> bool:
        try:
            return bool(self.state.kv_get("recorder.done"))
        except Exception:
            return False

    def run(self) -> int:
        start = self.clock()
        last_flush = start
        q: queue.Queue = queue.Queue()
        stop = threading.Event()
        thread: threading.Thread | None = None
        retry_at = 0.0
        self.hb.force(phase="starting", stream_id=self.stream_id)
        log.info("chat collector started for %s (bucket %ds)", self.stream_id, self.bucket_s)
        try:
            while True:
                now = self.clock()
                self.hb.tick(phase="collecting", stream_id=self.stream_id, msgs=self.total, failures=self.failures)
                if self._done():
                    log.info("recorder done: stopping chat collector (%d messages)", self.total)
                    return 0
                if self.max_seconds is not None and now - start >= self.max_seconds:
                    return 0
                if thread is None and now >= retry_at:
                    stop = threading.Event()
                    q = queue.Queue()
                    thread = threading.Thread(target=self._reader, args=(q, stop), daemon=True, name="chat-reader")
                    thread.start()
                try:
                    kind, val = q.get(timeout=0.25 if thread is not None else 0.05)
                except queue.Empty:
                    kind, val = None, None
                drained = 0
                while kind == "msg":
                    ts, author, text = val
                    arrival = self.clock()
                    self.add(float(ts) if ts else arrival, author, text)
                    self.failures = 0
                    self.last_msg_at = arrival
                    self._lag_sum += max(0.0, arrival - float(ts)) if ts else 0.0
                    drained += 1
                    if drained >= 500:
                        break
                    try:
                        kind, val = q.get_nowait()
                    except queue.Empty:
                        kind = None
                if kind in ("end", "err"):
                    stop.set()
                    thread = None
                    self.failures += 1
                    delay = self.backoff[min(self.failures - 1, len(self.backoff) - 1)]
                    retry_at = self.clock() + delay
                    if self.failures >= self.warn_after and not self.warned:
                        self.warned = True
                        log.warning("chat source keeps failing (%s); will keep retrying quietly every %.0fs",
                                    val if kind == "err" else "ended", delay)
                    else:
                        log.info("chat source %s (%s); retry in %.0fs", kind,
                                 val if kind == "err" else "stream ended?", delay)
                    elog.event("chat_reconnect", stream_id=self.stream_id, reason=kind, attempt=self.failures)
                now = self.clock()
                if now - last_flush >= self.poll_s:
                    self.flush(now)
                    last_flush = now
        finally:
            stop.set()
            try:
                self.flush()
            except Exception:
                log.exception("final chat flush failed")
            self.hb.force(phase="done", msgs=self.total)


# ---------------------------------------------------------------------- CLI
def _wait_for_stream(state: StateDB, hb: Heartbeat, poll_s: float = 3.0) -> tuple[str, str]:
    while True:
        hb.tick(phase="waiting")
        vid, sid = state.kv_get("recorder.video_id"), state.kv_get("stream.current")
        if vid and sid and not state.kv_get("recorder.done"):
            return str(vid), str(sid)
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clipper chat", description="Collect live-chat aggregates.")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--stream-id", default=None)
    ap.add_argument("--max-seconds", type=float, default=None, help="stop after N seconds (testing)")
    args = ap.parse_args(argv)
    elog.setup_logging("chat")
    for noisy in ("httpx", "httpcore", "hpack", "h2", "pytchat"):  # one INFO line per poll otherwise
        logging.getLogger(noisy).setLevel(logging.WARNING)
    crash.install_crash_handler("chat")
    cfg = load_config()
    if not bool(_opt(cfg.chat, "enabled", True)):
        log.info("chat collection disabled (chat.enabled: false)")
        return 0
    paths.DATA.ensure()
    state = StateDB()
    try:
        vid, sid = args.video_id, args.stream_id
        if not vid or not sid:
            hb = Heartbeat("chat", interval=5.0)
            wv, ws = _wait_for_stream(state, hb)
            vid, sid = vid or wv, sid or ws
        return Collector(cfg, state, vid, sid, max_seconds=args.max_seconds).run()
    finally:
        state.close()


if __name__ == "__main__":
    sys.exit(main())
