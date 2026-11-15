"""Penalties: menu/loading, radio masking, repeats."""
from __future__ import annotations

import pytest

from clipper.models import Candidate
from clipper.scoring import Window, activity_curve, score_chunk, score_window
from clipper.scoring.penalties import menu_loading_share
from tests.unit.test_scoring_synth import Synth, make_chunk


def test_menu_loading_window_penalised(cfg, state):
    s = Synth(300).gunfire(240, 250).loading(240, 270)
    a, v = s.build()
    act = activity_curve(300, a, v, cfg)
    assert menu_loading_share(v, 240, 270) == 1.0
    c = score_window(Window(240, 270, 242), make_chunk(), "test-stream", a, v, act, cfg, state)
    assert c.penalties["menu_loading"] == cfg.scoring.penalties.menu_loading
    assert "menu / loading screen" in c.events
    a2, v2 = Synth(300).gunfire(240, 250).build()
    clean = score_window(Window(240, 270, 242), make_chunk(), "test-stream", a2, v2,
                         activity_curve(300, a2, v2, cfg), cfg, state)
    assert "menu_loading" not in clean.penalties
    assert c.signal_score < clean.signal_score - 25


def test_static_cv_only_counts_as_menu(cfg, state):
    s = Synth(300, ai_ok=False).gunfire(240, 250)
    s.static[(s.mt >= 240) & (s.mt < 270)] = True
    a, v = s.build()
    c = score_window(Window(240, 270, 242), make_chunk(), "test-stream", a, v,
                     activity_curve(300, a, v, cfg), cfg, state)
    assert "menu_loading" in c.penalties


def test_loading_screen_produces_no_candidate(cfg, state):
    a, v = Synth(300).loading(240, 270).bang(250).build()
    assert score_chunk(make_chunk(), a, v, cfg, state) == []


def test_radio_masking_lowers_score(cfg, state):
    a, v = Synth(300).gunfire(75, 85).build()
    clean = score_chunk(make_chunk(), a, v, cfg, state)[0]
    a2, v2 = Synth(300).gunfire(75, 85).radio_on(60, 120).build()
    radio = score_chunk(make_chunk(idx=1, start_s=10_000), a2, v2, cfg, state)[0]
    assert radio.penalties["radio"] > 10
    assert radio.signal_score < clean.signal_score - 10
    assert radio.signal_score == pytest.approx(clean.signal_score - radio.penalties["radio"], abs=0.1)
    assert radio.signals["audio_spike"] == clean.signals["audio_spike"]   # raw signal kept for tuning
    assert radio.radio_ranges and radio.radio_ranges[0][0] >= 10_000 + 60   # stream seconds
    assert "radio playing" in radio.events


def test_repeat_near_selected_clip(cfg, state):
    prev = Candidate(id="test-stream-0-700", chunk_idx=0, start_s=64, peak_s=70, end_s=90, track="gameplay",
                     final_score=80, route="publish")
    state.save_candidate(prev, "test-stream")
    a, v = Synth(300).gunfire(75, 85).build()
    c = score_chunk(make_chunk(idx=1), a, v, cfg, state)[0]
    assert c.penalties["repeat"] == cfg.scoring.penalties.repeat
    assert any(e.startswith("repeat") for e in c.events)


def test_dropped_neighbour_is_not_a_repeat(cfg, state):
    prev = Candidate(id="old", chunk_idx=0, start_s=64, peak_s=70, end_s=90, track="gameplay",
                     final_score=20, route="drop")
    state.save_candidate(prev, "test-stream")
    a, v = Synth(300).gunfire(75, 85).build()
    assert "repeat" not in score_chunk(make_chunk(idx=1), a, v, cfg, state)[0].penalties


def test_repeat_within_chunk_hits_the_weaker(cfg, state):
    a, v = Synth(300).gunfire(75, 85, action=10).gunfire(115, 125, action=7).build()
    cands = score_chunk(make_chunk(), a, v, cfg, state)
    assert len(cands) == 2
    first, second = cands
    assert "repeat" not in first.penalties
    assert second.penalties.get("repeat") == cfg.scoring.penalties.repeat


def test_far_apart_moments_are_not_repeats(cfg, state):
    s = Synth(300).gunfire(30, 40).gunfire(200, 210)
    s.set_frames(150, 300, scene_tag="beach")
    a, v = s.build()
    cands = score_chunk(make_chunk(), a, v, cfg, state)
    assert len(cands) == 2 and all("repeat" not in c.penalties for c in cands)


def test_same_scene_tag_within_scene_window_is_repeat(cfg, state):
    prev = Candidate(id="prev-scene", chunk_idx=0, start_s=0, peak_s=0, end_s=30, track="cutscene",
                     final_score=80, route="publish", scene_tag="city street")
    state.save_candidate(prev, "test-stream")
    a, v = Synth(300).gunfire(100, 110).build()           # 100 s later: outside repeat_window_s (60)
    c = score_chunk(make_chunk(idx=1), a, v, cfg, state)[0]
    assert c.scene_tag == "city street"
    assert "same scene" in " ".join(c.events)
    assert c.penalties["repeat"] == cfg.scoring.penalties.repeat
