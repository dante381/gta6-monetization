"""Gameplay track: activity peaks, windows, signals, ai_ok=False reweighting."""
from __future__ import annotations

import numpy as np
import pytest

from clipper.scoring import activity_curve, detect_windows, find_peaks, score_chunk, weighted_score
from tests.unit.test_scoring_synth import Synth, make_chunk


def _gp_weights(cfg):
    return {k: float(v) for k, v in cfg.scoring.gameplay_weights.items()}


def test_quiet_chunk_has_no_candidates(cfg, state):
    a, v = Synth(300).build()
    assert score_chunk(make_chunk(), a, v, cfg, state) == []


def test_gunfire_and_motion_give_high_gameplay_candidate(cfg, state):
    a, v = Synth(300).gunfire(75, 85).build()
    cands = score_chunk(make_chunk(), a, v, cfg, state)
    assert len(cands) == 1
    c = cands[0]
    assert c.track == "gameplay"
    assert 75 <= c.peak_s < 85
    assert c.start_s == pytest.approx(c.peak_s - cfg.scoring.pre_roll_s)
    assert cfg.scoring.clip_min_s <= c.duration_s <= cfg.scoring.clip_max_s
    assert c.signal_score >= 60
    assert c.signals["audio_spike"] > 0.9 and c.signals["sound_type"] > 0.9
    assert c.signals["duration"] == 1.0
    assert "gunfire / explosion bursts" in c.events
    assert not c.penalties


def test_hud_wanted_stars_raise_score(cfg, state):
    a, v = Synth(300).gunfire(75, 85).build()
    base = score_chunk(make_chunk(), a, v, cfg, state)[0]
    s = Synth(300).gunfire(75, 85)
    s.set_frames(75, 300, wanted_stars=5)
    s.set_frames(75, 78, hud_events=["wanted_up"])
    a, v = s.build()
    c = score_chunk(make_chunk(idx=1, start_s=1000), a, v, cfg, state)[0]
    assert c.signals["hud_events"] == 1.0
    assert c.signal_score > base.signal_score + 15
    assert any("wanted level up to 5 stars" in e for e in c.events)


def test_mission_passed_hud(cfg, state):
    s = Synth(300).gunfire(75, 80)
    s.set_frames(78, 82, hud_events=["mission_passed"])
    a, v = s.build()
    c = score_chunk(make_chunk(), a, v, cfg, state)[0]
    assert c.signals["hud_events"] == 1.0
    assert "mission passed" in c.events


def test_single_bang_scores_low_duration(cfg, state):
    a, v = Synth(300).bang(100).build()
    cands = score_chunk(make_chunk(), a, v, cfg, state)
    assert len(cands) == 1
    c = cands[0]
    assert c.signals["duration"] < 0.3
    sustained = score_chunk(make_chunk(idx=2, start_s=5000), *Synth(300).gunfire(100, 110).build(), cfg, state)[0]
    assert sustained.signal_score > c.signal_score


def test_peaks_respect_min_gap(cfg):
    curve = np.zeros(100)
    curve[[10, 15, 40, 70]] = [0.9, 0.8, 0.7, 0.6]
    assert find_peaks(curve, 20, 0.25) == [10, 40, 70]


def test_window_extends_while_activity_high(cfg):
    a, v = Synth(300).gunfire(75, 105).build()
    act = activity_curve(300, a, v, cfg)
    w = detect_windows(act, v, cfg)[0]
    assert w.ws == pytest.approx(69)
    assert w.we == pytest.approx(105)                              # ends when the action ends
    a, v = Synth(300).gunfire(75, 140).build()
    w = detect_windows(activity_curve(300, a, v, cfg), v, cfg)[0]
    assert w.we - w.ws == pytest.approx(cfg.scoring.clip_max_s)   # capped at clip_max_s


def test_window_start_clamped_to_stream_zero(cfg, state):
    a, v = Synth(300).gunfire(1, 8).build()
    c = score_chunk(make_chunk(), a, v, cfg, state)[0]
    assert c.start_s == 0.0
    assert c.duration_s >= cfg.scoring.clip_min_s


def test_window_may_start_in_previous_chunk(cfg, state):
    a, v = Synth(300, idx=3).gunfire(2, 9).build()
    c = score_chunk(make_chunk(idx=3, start_s=900), a, v, cfg, state)[0]
    assert c.start_s == pytest.approx(c.peak_s - 6) and c.start_s < 900
    assert c.chunk_idx == 3
    assert c.id == f"test-stream-3-{int(c.peak_s * 10)}"


def test_ai_unavailable_reweights_over_measurable_signals(cfg, state):
    a, v = Synth(300, ai_ok=False).gunfire(75, 85).build()
    c = score_chunk(make_chunk(), a, v, cfg, state)[0]
    assert "hud_events" not in c.signals
    w = _gp_weights(cfg)
    raw_sum = sum(w[k] * c.signals[k] for k in c.signals)          # without rescaling (max 80)
    avail_w = sum(w[k] for k in c.signals)
    assert avail_w == 100 - w["hud_events"] - w["chat"]               # no Claude, no chat data
    assert c.signal_score == pytest.approx(100 * raw_sum / avail_w, abs=0.05)
    assert c.signal_score > raw_sum + 10
    assert c.signal_score >= 60          # not unfairly capped without Claude


def test_weighted_score_helper():
    w = {"a": 30, "b": 70}
    assert weighted_score({"a": 1.0, "b": 0.0}, w) == pytest.approx(30)
    assert weighted_score({"a": 1.0, "b": 0.0}, w, {"a"}) == pytest.approx(100)
    assert weighted_score({}, w, set()) == 0.0
