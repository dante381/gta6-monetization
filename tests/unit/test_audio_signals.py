"""Audio analysis on generated numpy signals (no ffmpeg)."""
from __future__ import annotations

import numpy as np
import pytest

from clipper.analysis import audio as A

RATE = 16000


def _noise(sec, amp, rng):
    return (amp * (rng.random(int(sec * RATE)) * 2 - 1)).astype(np.float32)


def _chord(sec, amp, t0=0.0, freqs=(440.0, 554.37, 659.25)):
    t = t0 + np.arange(int(sec * RATE)) / RATE
    return (amp * sum(np.sin(2 * np.pi * f * t) for f in freqs)).astype(np.float32)


def test_hop_grid_and_partial_tail():
    x = np.zeros(int(10.2 * RATE), dtype=np.float32)
    f, raw = A.features_from_samples(x, RATE, None, return_raw=True)
    assert len(f.t) == 20                       # 0.2 s tail < half a hop is dropped
    assert np.allclose(f.t[:3], [0.25, 0.75, 1.25])
    assert all(len(getattr(f, k)) == 20 for k in ("loudness_db", "loud_z", "spike", "transient", "music", "swell"))
    x2 = np.zeros(int(10.3 * RATE), dtype=np.float32)       # 0.3 s tail >= half a hop is kept
    assert len(A.features_from_samples(x2, RATE, None).t) == 21


def test_block_size_does_not_change_results(cfg):
    rng = np.random.default_rng(1)
    x = np.concatenate([_noise(20, 0.01, rng), _noise(3, 0.8, rng), _chord(25, 0.15), _noise(10, 0.01, rng)])
    ex1 = A.HopExtractor(RATE, cfg)
    ex1.push(x)
    r1 = ex1.finish()
    ex2 = A.HopExtractor(RATE, cfg)
    for i in range(0, x.size, 12345):           # odd block size, not aligned to hops
        ex2.push(x[i:i + 12345])
    r2 = ex2.finish()
    for k in ("loudness_db", "flatness", "stability", "onset_db", "burst_db"):
        assert np.allclose(getattr(r1, k), getattr(r2, k), atol=1e-6), k


def test_silence_gives_nothing():
    x = np.zeros(60 * RATE, dtype=np.float32)
    f = A.features_from_samples(x, RATE, None)
    assert f.spike.max() == 0 and f.transient.max() == 0 and f.swell.max() == 0
    assert not f.music.any() and f.radio_ranges == []
    assert np.all(np.isfinite(f.loud_z))


def test_silence_to_noise_floor_is_not_a_spike():
    rng = np.random.default_rng(2)
    x = np.concatenate([_noise(30, 0.01, rng), np.zeros(30 * RATE, np.float32), _noise(30, 0.01, rng)])
    f = A.features_from_samples(x, RATE, None)
    assert f.spike.max() < 0.05
    assert f.transient.max() < 0.2


def test_burst_spike_and_transient():
    rng = np.random.default_rng(3)
    x = np.concatenate([_noise(30, 0.01, rng), _noise(1.0, 0.9, rng), _noise(30, 0.01, rng)])
    f = A.features_from_samples(x, RATE, None)
    s = f.at(30, 31)
    assert f.spike[s].max() > 0.9
    assert f.transient[s].max() > 0.5       # flat uniform noise has low crest; real bangs score ~1
    assert max(f.spike[f.at(0, 29)].max(), f.spike[f.at(33, 61)].max()) < 0.05
    assert f.loud_z[s].max() > 5


def test_tonal_stretch_is_music_noise_is_not(cfg):
    rng = np.random.default_rng(4)
    x = np.concatenate([_noise(20, 0.01, rng), _chord(40, 0.15), _noise(40, 0.3, rng), _chord(10, 0.15)])
    f = A.features_from_samples(x, RATE, cfg)
    assert len(f.radio_ranges) == 1             # the 10 s chord is shorter than radio_min_s
    a, b = f.radio_ranges[0]
    assert abs(a - 20) <= 1.0 and abs(b - 60) <= 1.0
    assert not f.music[f.at(61, 100)].any()
    assert f.transient[f.at(21, 59)].max() < 0.1   # tones are not transients


def test_swell_rises_only_when_level_rises(cfg):
    rng = np.random.default_rng(5)
    flat = _chord(30, 0.1, freqs=(220.0, 330.0))
    ramp_t = np.arange(30 * RATE) / RATE
    ramp = (np.sin(2 * np.pi * 220 * ramp_t) + 0.5 * np.sin(2 * np.pi * 330 * ramp_t)) * (0.05 * 10 ** (ramp_t / 30 * 1.2))
    x = np.concatenate([_noise(20, 0.01, rng), flat, _noise(10, 0.01, rng), ramp.astype(np.float32),
                        _noise(10, 0.01, rng)])
    f = A.features_from_samples(x, RATE, cfg)
    assert f.music[f.at(22, 49)].all() and f.music[f.at(62, 89)].all()
    assert f.swell[f.at(22, 49)].max() < 0.2           # steady level
    assert f.swell[f.at(70, 89)].mean() > 0.6           # 24 dB crescendo
    assert f.swell[~f.music].max() == 0


def test_baseline_history_changes_loud_z(cfg):
    rng = np.random.default_rng(6)
    x = np.concatenate([_noise(30, 0.01, rng), _noise(1.0, 0.9, rng), _noise(29, 0.01, rng)])
    own = A.features_from_samples(x, RATE, cfg)
    loud_hist = A.Baseline(history_db=np.full(600, -10.0))
    other = A.features_from_samples(x, RATE, cfg, baseline=loud_hist)
    assert other.loud_z.mean() < own.loud_z.mean() - 3
    assert other.spike.max() < own.spike.max()      # burst is not above a loud stream's baseline


def test_merge_and_stream_ranges():
    assert A.merge_ranges([(5, 8), (1, 3), (2.5, 4), (9, 9)]) == [(1.0, 4.0), (5.0, 8.0)]
    assert A.merge_ranges([(1, 3), (3.5, 5)], gap_s=1.0) == [(1.0, 5.0)]
    assert A.to_stream_ranges([(10, 20), (19, 30)], 300.0) == [(310.0, 330.0)]
    assert A.clip_ranges([(0, 10), (20, 30)], 5, 25) == [(5.0, 10.0), (20.0, 25.0)]


def test_trailing_tonal_run_carries_into_next_chunk(cfg):
    x = _chord(12, 0.15)                      # 12 s < radio_min_s on its own
    alone = A.features_from_samples(x, RATE, cfg)
    assert not alone.music.any()
    carried = A.features_from_samples(x, RATE, cfg, baseline=A.Baseline(tonal_tail_s=15.0))
    assert carried.music.sum() >= 22 and carried.radio_ranges[0][0] == 0.0


@pytest.mark.parametrize("n", [0, 100])
def test_tiny_inputs(n):
    f = A.features_from_samples(np.zeros(n, np.float32), RATE, None)
    assert len(f.t) == 0 and f.radio_ranges == []
