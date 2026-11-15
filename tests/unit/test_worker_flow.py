"""Worker orchestration with stubbed stages: ordering, deferral across chunks, cleanup rule, drain."""
from __future__ import annotations

from pathlib import Path

from clipper.models import Candidate, Chunk, ClipMeta, ClipSidecar
from clipper.worker import Worker


def _mk_chunks(state, tmp_path, n=3, dur=300.0):
    sid = state.current_stream()
    out = []
    for i in range(n):
        p = tmp_path / f"chunk_{i:05d}.ts"
        p.write_bytes(b"x")
        c = Chunk(idx=i, path=str(p), start_s=i * dur, duration_s=dur, stream_id=sid)
        state.add_chunk(c)
        out.append(c)
    return out


class Stages:
    def __init__(self, cands_by_chunk):
        self.cands_by_chunk = cands_by_chunk
        self.order = []
        self.produced = []
        self.enqueued = []

    def audio(self, chunk, cfg, state, on_tick=None):
        self.order.append(("audio", chunk.idx))
        return object()

    def video(self, chunk, cfg, on_tick=None):
        self.order.append(("video", chunk.idx))
        return type("V", (), {"ai_ok": True})()

    def score(self, chunk, audio, visual, cfg, state, on_tick=None):
        self.order.append(("score", chunk.idx))
        cands = self.cands_by_chunk.get(chunk.idx, [])
        for c in cands:
            state.save_candidate(c, chunk.stream_id)
        return cands

    def produce(self, cand, chunks, cfg, on_tick=None):
        self.produced.append((cand.id, [c.idx for c in chunks], cand.start_s, cand.end_s))
        return ClipSidecar(clip_id=cand.id, clip_path="x.mp4", track=cand.track, route=cand.route,
                           final_score=cand.final_score, signal_score=cand.signal_score, title="t",
                           description="", hashtags=[], hook="", spoiler=False, duration_s=cand.duration_s,
                           start_s=cand.start_s, end_s=cand.end_s)

    def enqueue(self, sidecar, state, cfg):
        self.enqueued.append(sidecar.clip_id)

    def publish(self, cfg, state, **kw):
        return []


def _worker(cfg, state, st):
    return Worker(cfg, state, analyze_audio=st.audio, analyze_video=st.video, score_chunk=st.score,
                  produce_clip=st.produce, publish_tick=st.publish, enqueue=st.enqueue, idle_sleep_s=0)


def cand(cid, chunk_idx, start, end, route="publish", score=80):
    return Candidate(id=cid, chunk_idx=chunk_idx, start_s=start, peak_s=start + 6, end_s=end,
                     track="gameplay", final_score=score, route=route, meta=ClipMeta(title="t"))


def test_boundary_clip_uses_both_chunks_when_both_on_disk(state, cfg, tmp_path):
    _mk_chunks(state, tmp_path, n=2)
    st = Stages({0: [cand("a", 0, 280.0, 320.0)]})
    w = _worker(cfg, state, st)
    sid = state.current_stream()
    w.process_chunk(state.next_chunk_to_process(sid))
    assert w.produce_pending(sid) == 1
    assert st.produced == [("a", [0, 1], 280.0, 320.0)]


def test_deferral_when_footage_missing(state, cfg, tmp_path):
    sid = state.current_stream()
    p = tmp_path / "c0.ts"; p.write_bytes(b"x")
    state.add_chunk(Chunk(idx=0, path=str(p), start_s=0, duration_s=300, stream_id=sid))
    st = Stages({0: [cand("a", 0, 280.0, 320.0)]})
    w = _worker(cfg, state, st)
    w.process_chunk(state.next_chunk_to_process(sid))
    assert w.produce_pending(sid) == 0 and st.produced == []   # runs past newest footage -> deferred
    # chunk 1 arrives
    p1 = tmp_path / "c1.ts"; p1.write_bytes(b"x")
    state.add_chunk(Chunk(idx=1, path=str(p1), start_s=300, duration_s=300, stream_id=sid))
    assert w.produce_pending(sid) == 1
    assert st.produced[0][1] == [0, 1]
    assert st.enqueued == ["a"]


def test_final_clamps_clip_at_stream_end(state, cfg, tmp_path):
    sid = state.current_stream()
    p = tmp_path / "c0.ts"; p.write_bytes(b"x")
    state.add_chunk(Chunk(idx=0, path=str(p), start_s=0, duration_s=100, stream_id=sid))
    st = Stages({0: [cand("a", 0, 80.0, 120.0)]})
    w = _worker(cfg, state, st)
    w.process_chunk(state.next_chunk_to_process(sid))
    assert w.produce_pending(sid, final=True) == 1
    _, _, s, e = st.produced[0]
    assert e == 100.0 and e - s >= cfg.scoring.clip_min_s


def test_dropped_candidates_never_produced(state, cfg, tmp_path):
    _mk_chunks(state, tmp_path, n=1)
    st = Stages({0: [cand("d", 0, 10, 40, route="drop", score=30)]})
    w = _worker(cfg, state, st)
    sid = state.current_stream()
    w.process_chunk(state.next_chunk_to_process(sid))
    assert w.produce_pending(sid, final=True) == 0


def test_cleanup_waits_for_next_chunk_processed(state, cfg, tmp_path):
    chunks = _mk_chunks(state, tmp_path, n=3)
    sid = state.current_stream()
    st = Stages({})
    w = _worker(cfg, state, st)
    w.process_chunk(state.next_chunk_to_process(sid))           # chunk 0 processed
    assert w.cleanup(sid) == 0 and Path(chunks[0].path).exists()
    w.process_chunk(state.next_chunk_to_process(sid))           # chunk 1 processed
    assert w.cleanup(sid) == 1                                   # chunk 0 gone, chunk 1 kept (2 not processed)
    assert not Path(chunks[0].path).exists() and Path(chunks[1].path).exists()
    assert state.get_chunk(sid, 0).status == "deleted"


def test_keep_chunks_hold_blocks_cleanup(state, cfg, tmp_path):
    chunks = _mk_chunks(state, tmp_path, n=2)
    sid = state.current_stream()
    w = _worker(cfg, state, Stages({}))
    for _ in range(2):
        w.process_chunk(state.next_chunk_to_process(sid))
    state.set_keep_chunks(True)
    assert w.cleanup(sid) == 0 and Path(chunks[0].path).exists()
    state.set_keep_chunks(False)
    assert w.cleanup(sid) == 1


def test_unproduced_clip_blocks_deletion(state, cfg, tmp_path):
    chunks = _mk_chunks(state, tmp_path, n=3)
    sid = state.current_stream()
    st = Stages({0: [cand("a", 0, 280.0, 320.0)]})
    w = _worker(cfg, state, st)
    w.process_chunk(state.next_chunk_to_process(sid))
    w.process_chunk(state.next_chunk_to_process(sid))
    assert w.cleanup(sid) == 0                                   # clip a not cut yet
    assert w.produce_pending(sid) == 1
    assert w.cleanup(sid) == 1


def test_run_exits_after_stream_end_and_drains(state, cfg, tmp_path):
    _mk_chunks(state, tmp_path, n=2)
    sid = state.current_stream()
    state.kv_set("recorder.done", True)
    st = Stages({0: [cand("a", 0, 100.0, 140.0)], 1: [cand("b", 1, 400.0, 440.0)]})
    w = _worker(cfg, state, st)
    assert w.run(max_loops=20) == 0
    assert sorted(st.enqueued) == ["a", "b"]
    assert [c.status for c in state.chunks(sid)] == ["deleted", "deleted"]
    assert [o for o in st.order if o[0] == "audio"] == [("audio", 0), ("audio", 1)]


def test_publish_error_does_not_stop_worker(state, cfg, tmp_path):
    _mk_chunks(state, tmp_path, n=1)
    state.kv_set("recorder.done", True)
    st = Stages({})

    def boom(cfg, state, **kw):
        raise RuntimeError("youtube exploded")

    w = Worker(cfg, state, analyze_audio=st.audio, analyze_video=st.video, score_chunk=st.score,
               produce_clip=st.produce, publish_tick=boom, enqueue=st.enqueue, idle_sleep_s=0)
    assert w.run(max_loops=10) == 0
