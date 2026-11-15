"""Processing worker: one chunk at a time.

Loop:
  1. next recorded chunk -> audio analysis (CPU) -> vision (GPU decode, then ONE claude call)
     -> scoring (+judge) -> mark processed
  2. produce clips whose footage is fully on disk (a candidate running past the newest chunk
     waits for the next chunk; at stream end it is clamped to the footage available)
  3. queue produced clips for publishing, run a publish tick
  4. delete chunks that are no longer needed:
       chunk N is deleted when it is processed, all clips touching it are cut, AND chunk N+1
       is processed (or the stream has ended) -- unless the supervisor set keep_chunks.
  5. stream ended and nothing left -> drain the publish queue -> exit 0

Stages never overlap (RAM cap: no claude call while a GPU ffmpeg job runs).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from . import config as config_mod
from . import crash, heartbeat, paths
from . import log as logmod
from .env import apply_to_current_process
from .models import Candidate, Chunk
from .state import StateDB

log = logging.getLogger("worker")


class Worker:
    def __init__(self, cfg, state: StateDB, hb: heartbeat.Heartbeat | None = None, *,
                 analyze_audio=None, analyze_video=None, score_chunk=None, produce_clip=None,
                 publish_tick=None, enqueue=None, idle_sleep_s: float = 5.0):
        self.cfg = cfg
        self.state = state
        self.hb = hb or heartbeat.Heartbeat("worker", interval=cfg.supervisor.heartbeat_s)
        self.idle_sleep_s = idle_sleep_s
        # stages are injectable for tests; default to the real modules (imported lazily)
        self._analyze_audio = analyze_audio
        self._analyze_video = analyze_video
        self._score_chunk = score_chunk
        self._produce_clip = produce_clip
        self._publish_tick = publish_tick
        self._enqueue = enqueue

    # ------------------------------------------------------------------ stage resolution
    def _stage(self, name: str):
        fn = getattr(self, f"_{name}")
        if fn is not None:
            return fn
        if name == "analyze_audio":
            from .analysis.audio import analyze_audio as fn
        elif name == "analyze_video":
            from .analysis.vision import analyze_video as fn
        elif name == "score_chunk":
            from .scoring import score_chunk as fn
        elif name == "produce_clip":
            from .production import produce_clip as fn
        elif name == "publish_tick":
            from .publish import publish_tick as fn
        elif name == "enqueue":
            from .publish import enqueue as fn
        setattr(self, f"_{name}", fn)
        return fn

    def tick(self, **info) -> None:
        self.hb.tick(**info)

    # ------------------------------------------------------------------ main loop
    def run(self, max_loops: int | None = None) -> int:
        loops = 0
        self.hb.force(stage="start")
        while max_loops is None or loops < max_loops:
            loops += 1
            self.tick(stage="loop")
            sid = self.state.current_stream()
            did_work = False
            if sid:
                chunk = self.state.next_chunk_to_process(sid)
                if chunk is not None:
                    self.process_chunk(chunk)
                    did_work = True
                did_work |= self.produce_pending(sid) > 0
                self.cleanup(sid)
            self.publish()
            if sid and self.stream_finished(sid):
                self.produce_pending(sid, final=True)
                self.cleanup(sid, final=True)
                self.drain_publish()
                logmod.event("worker_done", stream_id=sid)
                log.info("stream %s finished and drained; worker exiting", sid)
                return 0
            if not did_work:
                self._sleep(self.idle_sleep_s)
        return 0

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            time.sleep(min(1.0, max(0.0, end - time.monotonic())))
            self.tick()

    # ------------------------------------------------------------------ chunk processing
    def process_chunk(self, chunk: Chunk) -> list[Candidate]:
        sid = chunk.stream_id
        log.info("processing chunk %d (%s)", chunk.idx, chunk.path)
        t0 = time.monotonic()
        self.state.update_chunk(sid, chunk.idx, status="processing")
        self.tick(stage="audio", chunk=chunk.idx)
        audio = self._stage("analyze_audio")(chunk, self.cfg, self.state, on_tick=self.tick)
        self.tick(stage="vision", chunk=chunk.idx)
        visual = self._stage("analyze_video")(chunk, self.cfg, on_tick=self.tick)
        self.tick(stage="scoring", chunk=chunk.idx)
        cands = self._stage("score_chunk")(chunk, audio, visual, self.cfg, self.state, on_tick=self.tick)
        self.state.update_chunk(sid, chunk.idx, status="processed", processed_at=time.time())
        n_sel = sum(1 for c in cands if c.route != "drop")
        logmod.event("chunk_processed", stream_id=sid, chunk=chunk.idx, candidates=len(cands),
                     selected=n_sel, secs=round(time.monotonic() - t0, 1), ai_ok=getattr(visual, "ai_ok", None))
        return cands

    # ------------------------------------------------------------------ production
    def _footage_end(self, sid: str) -> float:
        chunks = [c for c in self.state.chunks(sid) if c.status != "deleted"]
        return max((c.end_s for c in chunks), default=0.0)

    def produce_pending(self, sid: str, final: bool = False) -> int:
        """Produce every selected, unproduced candidate whose footage is on disk."""
        produced = 0
        pending = [c for c in self.state.candidates(sid, produced=False) if c.route != "drop"]
        if not pending:
            return 0
        footage_end = self._footage_end(sid)
        min_len = float(self.cfg.scoring.clip_min_s)
        for cand in sorted(pending, key=lambda c: -c.final_score):
            self.tick(stage="produce", cand=cand.id)
            if cand.end_s > footage_end + 0.05:
                if not final:
                    continue  # wait for the next chunk
                cand.end_s = footage_end
                if cand.end_s - cand.start_s < min_len:
                    cand.start_s = max(0.0, cand.end_s - min_len)
            chunks = self.state.chunks_covering(sid, cand.start_s, cand.end_s)
            if not chunks:
                log.warning("no footage left for candidate %s; skipping", cand.id)
                self.state.mark_produced(cand.id)
                continue
            # clamp to footage that still exists (earlier chunk may already be gone)
            cand.start_s = max(cand.start_s, chunks[0].start_s)
            cand.end_s = min(cand.end_s, chunks[-1].end_s)
            if cand.end_s - cand.start_s < 5:
                log.warning("candidate %s too short after clamping; skipping", cand.id)
                self.state.mark_produced(cand.id)
                continue
            sidecar = self._stage("produce_clip")(cand, chunks, self.cfg, on_tick=self.tick)
            self._stage("enqueue")(sidecar, self.state, self.cfg)
            self.state.save_candidate(cand, sid, produced=True)
            produced += 1
        return produced

    # ------------------------------------------------------------------ publishing
    def publish(self) -> None:
        try:
            self._stage("publish_tick")(self.cfg, self.state)
        except Exception:  # publishing problems must never stop recording/clipping
            log.exception("publish tick failed")
            logmod.event("publish_error")

    def drain_publish(self, max_rounds: int = 3) -> None:
        for _ in range(max_rounds):
            if not self.state.queue_items(status="queued"):
                return
            self.publish()
            self.tick(stage="drain")

    # ------------------------------------------------------------------ chunk cleanup
    def _chunk_clips_cut(self, sid: str, chunk: Chunk) -> bool:
        for c in self.state.candidates(sid, produced=False):
            if c.route == "drop":
                continue
            if c.start_s < chunk.end_s and c.end_s > chunk.start_s:
                return False
        return True

    def cleanup(self, sid: str, final: bool = False) -> int:
        if self.state.keep_chunks():
            return 0
        chunks = self.state.chunks(sid)
        by_idx = {c.idx: c for c in chunks}
        done = bool(self.state.kv_get("recorder.done", False))
        deleted = 0
        for c in chunks:
            if c.status != "processed":
                continue
            if not self._chunk_clips_cut(sid, c):
                continue
            self.state.mark_clips_cut(sid, c.idx)
            nxt = by_idx.get(c.idx + 1)
            next_ok = (nxt is not None and nxt.status in ("processed", "deleted")) or (final and done)
            if not next_ok:
                continue
            self._delete_chunk_files(sid, c)
            self.state.update_chunk(sid, c.idx, status="deleted")
            deleted += 1
        return deleted

    def _delete_chunk_files(self, sid: str, c: Chunk) -> None:
        try:
            Path(c.path).unlink(missing_ok=True)
        except OSError as e:
            log.warning("could not delete %s: %s", c.path, e)
        for base in (paths.DATA.frames, paths.DATA.grids):
            shutil.rmtree(base / sid / str(c.idx), ignore_errors=True)
        logmod.event("chunk_deleted", stream_id=sid, chunk=c.idx)

    # ------------------------------------------------------------------ end of stream
    def stream_finished(self, sid: str) -> bool:
        if not self.state.kv_get("recorder.done", False):
            return False
        if self.state.next_chunk_to_process(sid) is not None:
            return False
        return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clipper work")
    ap.add_argument("--max-loops", type=int, default=None)
    a = ap.parse_args(argv)
    apply_to_current_process()
    logmod.setup_logging("worker")
    crash.install_crash_handler("worker")
    paths.DATA.ensure()
    cfg = config_mod.load_config()
    state = StateDB()
    return Worker(cfg, state).run(max_loops=a.max_loops)
