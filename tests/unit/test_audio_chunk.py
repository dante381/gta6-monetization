"""analyze_audio on the synthetic fixture chunks (real ffmpeg decode, CPU only)."""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from clipper.analysis import audio as A
from clipper.ffmpeg import FfmpegError
from clipper.models import Chunk


def _chunk(path, idx=0, start_s=0.0, dur=300.0, stream="test-stream"):
    return Chunk(idx=idx, path=str(path), start_s=start_s, duration_s=dur, stream_id=stream)


def _win(f, ev, name, pad=0.0):
    a, b = ev[name]
    return f.at(a - pad, b + pad)


@pytest.fixture(scope="module")
def full(synthetic_chunk, tmp_path_factory):
    """Analyse the 300 s chunk once (own throwaway state) and time it."""
    from clipper.config import load_config
    from clipper.state import StateDB
    path, truth = synthetic_chunk
    db = StateDB(tmp_path_factory.mktemp("audio_state") / "state.db")
    ticks = []
    t0 = time.perf_counter()
    f = A.analyze_audio(_chunk(path), load_config(), db, on_tick=lambda: ticks.append(1))
    elapsed = time.perf_counter() - t0
    db.close()
    return f, truth["events"], elapsed, len(ticks)


def test_shapes_and_runtime(full):
    f, ev, elapsed, ticks = full
    print(f"\n300 s chunk analysed in {elapsed:.2f} s, {ticks} ticks, ranges={f.radio_ranges}")
    assert 595 <= len(f.t) <= 601
    assert f.hop_s == 0.5
    for k in ("loudness_db", "loud_z", "spike", "transient", "music", "swell"):
        assert len(getattr(f, k)) == len(f.t)
    for k in ("spike", "transient", "swell"):
        v = getattr(f, k)
        assert v.min() >= 0 and v.max() <= 1 and np.all(np.isfinite(v))
    assert f.music.dtype == bool
    assert elapsed < 15.0
    assert ticks >= 30                          # one per 10 s block


def test_spike_and_transient_peaks_in_action_windows(full):
    f, ev, *_ = full
    for name in ("spike", "transient"):
        v = getattr(f, name)
        peak_t = f.t[int(np.argmax(v))]
        in_action = any(ev[e][0] - 0.5 <= peak_t <= ev[e][1] + 0.5 for e in ("gunfire", "explosion"))
        assert in_action, (name, peak_t)
        assert v[_win(f, ev, "gunfire")].max() > 0.9
        assert v[_win(f, ev, "explosion")].max() > 0.9
        assert v[_win(f, ev, "loading", pad=0.5)].max() < 0.1
        assert v[_win(f, ev, "radio")].max() <= 0.5
    assert f.transient[_win(f, ev, "radio")].max() < 0.1
    assert f.transient[_win(f, ev, "cutscene")].max() < 0.1
    # quiet gameplay produces nothing
    assert f.spike[f.at(5, 70)].max() < 0.05 and f.transient[f.at(5, 70)].max() < 0.1
    # gunfire transient is sustained (duration signal), not a single hop
    assert f.transient[_win(f, ev, "gunfire")].mean() > 0.8


def test_radio_ranges_cover_radio_not_gunfire(full):
    f, ev, *_ = full
    r0, r1 = ev["radio"]
    covered = sum(max(0.0, min(b, r1) - max(a, r0)) for a, b in f.radio_ranges)
    assert covered >= 0.9 * (r1 - r0), f.radio_ranges
    g0, g1 = ev["gunfire"]
    assert all(b <= g0 or a >= g1 for a, b in f.radio_ranges)
    e0, e1 = ev["explosion"]
    assert all(b <= e0 or a >= e1 for a, b in f.radio_ranges)
    assert not f.music[f.at(0, 70)].any()
    assert not f.music[_win(f, ev, "loading")].any()
    # every flagged range is long enough
    assert all(b - a >= 20 for a, b in f.radio_ranges)
    # stream-seconds helper
    s = A.to_stream_ranges(f.radio_ranges, 1200.0)
    assert s[0][0] == pytest.approx(1200.0 + f.radio_ranges[0][0])


def test_swell_high_in_second_half_of_cutscene(full):
    f, ev, *_ = full
    c0, c1 = ev["cutscene"]
    mid = (c0 + c1) / 2
    late = f.swell[f.at(mid, c1)]
    print(f"\nswell late mean={late.mean():.2f} min={late.min():.2f}")
    assert late.mean() > 0.6
    assert f.swell[_win(f, ev, "radio")].max() < 0.2     # steady radio is not a swell
    assert f.swell[~f.music].max() == 0


def test_loud_z_relative_to_baseline(full):
    f, ev, *_ = full
    assert f.loud_z[_win(f, ev, "gunfire")].mean() > 3
    assert abs(np.median(f.loud_z[f.at(5, 70)])) < 1.5


def test_baseline_persists_between_chunks(synthetic_short, cfg, state):
    path, truth = synthetic_short
    cfg.analysis.radio_min_s = 4
    c0 = _chunk(path, idx=0, start_s=0.0, dur=60.0)
    c1 = _chunk(path, idx=1, start_s=60.0, dur=60.0)
    assert A.load_baseline(state, c0, cfg).history_db.size == 0
    f0 = A.analyze_audio(c0, cfg, state)
    kv = state.kv_get(A.KV_KEY)
    assert kv["stream_id"] == "test-stream" and set(kv["segments"]) == {"0"}
    assert 50 <= len(kv["segments"]["0"]["db"]) <= 58      # 1 s resolution, 6 s of silence dropped
    assert len(kv["segments"]["0"]["tail"]) == 20
    assert len(state.conn.execute("SELECT value FROM kv WHERE key=?", (A.KV_KEY,)).fetchone()[0]) < 4000

    bl1 = A.load_baseline(state, c1, cfg)
    assert bl1.history_db.size == len(kv["segments"]["0"]["db"]) and bl1.tail_db.size == 20
    f1 = A.analyze_audio(c1, cfg, state)
    kv = state.kv_get(A.KV_KEY)
    assert set(kv["segments"]) == {"0", "1"}
    # re-analysing chunk 1 (crash retry) replaces, never doubles
    A.analyze_audio(c1, cfg, state)
    assert set(state.kv_get(A.KV_KEY)["segments"]) == {"0", "1"}
    assert A.load_baseline(state, _chunk(path, idx=2), cfg).history_db.size == 2 * bl1.history_db.size

    ev = truth["events"]
    for f in (f0, f1):
        assert f.spike[_win(f, ev, "gunfire")].max() > 0.9
        assert f.transient[_win(f, ev, "explosion")].max() > 0.9
    # a different stream starts fresh
    assert A.load_baseline(state, _chunk(path, idx=1, stream="other"), cfg).history_db.size == 0


def test_loud_history_lowers_scores(synthetic_short, cfg, state):
    path, truth = synthetic_short
    c = _chunk(path, idx=5, dur=60.0)
    fresh = A.analyze_audio(c, cfg, None)
    state.kv_set(A.KV_KEY, {"v": 1, "stream_id": "test-stream", "res_s": 1.0,
                            "segments": {"4": {"db": [-5.0] * 600, "tail": [-5.0] * 20, "tonal_tail_s": 0}}})
    loud = A.analyze_audio(c, cfg, state)
    assert loud.loud_z.mean() < fresh.loud_z.mean() - 2
    assert loud.spike.max() < fresh.spike.max()


def test_decode_streams_with_small_memory(synthetic_chunk, cfg):
    path, _ = synthetic_chunk
    tracemalloc.start()
    try:
        f = A.analyze_audio(_chunk(path), cfg, None)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    whole_chunk = 300 * 16000 * 4            # bytes if the chunk were decoded at once
    print(f"\npeak traced memory {peak / 1e6:.1f} MB (whole decoded chunk would be {whole_chunk / 1e6:.1f} MB)")
    assert peak < 12e6
    assert len(f.t) >= 595


def test_missing_file_raises_ffmpeg_error(tmp_path, cfg):
    with pytest.raises(FfmpegError):
        A.analyze_audio(_chunk(tmp_path / "nope.ts"), cfg, None)


def test_file_without_audio_raises(tmp_path, cfg):
    p = tmp_path / "noaudio.txt"
    p.write_text("not media", encoding="utf-8")
    with pytest.raises(FfmpegError):
        A.analyze_audio(_chunk(p), cfg, None)
