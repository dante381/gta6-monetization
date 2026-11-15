"""Cutscene track: letterbox / Claude labels, trimming, newness, title text."""
from __future__ import annotations

from clipper.scoring import activity_curve, assign_track, cutscene_signals, score_chunk
from clipper.scoring.cutscene import cutscene_window
from tests.unit.test_scoring_synth import Synth, make_chunk


def _cutscene_synth(**kw):
    s = Synth(300, **kw).letterbox(150, 270).swell_up(150, 270)
    if s.ai_ok:
        s.set_frames(150, 270, scene_tag="rooftop")
    return s


def test_letterbox_gives_cutscene_track_and_trimmed_length(cfg, state):
    a, v = _cutscene_synth().build()
    cands = score_chunk(make_chunk(), a, v, cfg, state)
    cuts = [c for c in cands if c.track == "cutscene"]
    assert cuts, cands
    c = cuts[0]
    assert cfg.scoring.cutscene_min_s <= c.duration_s <= cfg.scoring.clip_max_s
    assert 150 <= c.start_s and c.end_s <= 270
    assert set(c.signals) == {"title_text", "spectacle", "newness", "music_swell"}
    assert c.signals["newness"] == 1.0
    assert c.scene_tag == "rooftop"
    assert c.meta.spoiler and "spoiler" in c.meta.title.lower()


def test_long_cutscene_trimmed_to_strongest_part(cfg):
    s = _cutscene_synth()
    s.gunfire(220, 228)                 # explosion-heavy stretch inside the cutscene
    s.set_frames(150, 270, kind="cutscene", letterbox=True)
    s.letter[(s.mt >= 150) & (s.mt < 270)] = True
    a, v = s.build()
    act = activity_curve(300, a, v, cfg)
    ws, we = cutscene_window(act, 220, 214, 259, cfg)
    assert ws <= 220 < we
    assert cfg.scoring.cutscene_min_s <= we - ws <= cfg.scoring.clip_max_s


def test_short_cutscene_padded_to_minimum(cfg):
    s = Synth(300).letterbox(100, 112)
    a, v = s.build()
    act = activity_curve(300, a, v, cfg)
    ws, we = cutscene_window(act, 100, 94, 114, cfg)
    assert we - ws == cfg.scoring.cutscene_min_s
    assert ws <= 100 and we >= 112


def test_claude_majority_cutscene_without_letterbox(cfg, state):
    s = Synth(300).bang(110)
    s.set_frames(96, 135, kind="cutscene")
    a, v = s.build()
    assert assign_track(v, 104, 124, cfg) == "cutscene"
    c = score_chunk(make_chunk(), a, v, cfg, state)[0]
    assert c.track == "cutscene"
    # without Claude, labels don't count (CV letterbox only)
    s2 = Synth(300, ai_ok=False).bang(110)
    s2.set_frames(96, 135, kind="cutscene")
    _, v2 = s2.build()
    assert assign_track(v2, 104, 124, cfg) == "gameplay"


def test_newness_decays_with_scene_count(cfg, state):
    a, v = _cutscene_synth().build()
    act = activity_curve(300, a, v, cfg)
    fresh = cutscene_signals(150, 195, 150, a, v, act, cfg, state)
    assert fresh.raw["newness"] == 1.0
    assert "new scene: rooftop" in fresh.events
    state.mark_scene("rooftop", 10)
    once = cutscene_signals(150, 195, 150, a, v, act, cfg, state)
    state.mark_scene("rooftop", 20)
    twice = cutscene_signals(150, 195, 150, a, v, act, cfg, state)
    assert once.raw["newness"] == 0.5
    assert twice.raw["newness"] == 0.25
    assert not any(e.startswith("new scene") for e in twice.events)


def test_title_text_signal(cfg, state):
    s = _cutscene_synth()
    s.set_frames(150, 160, title_text="VICE CITY")
    a, v = s.build()
    act = activity_curve(300, a, v, cfg)
    sr = cutscene_signals(150, 195, 150, a, v, act, cfg, state)
    assert sr.raw["title_text"] == 1.0
    assert 'title card: "VICE CITY"' in sr.events
    c = score_chunk(make_chunk(), a, v, cfg, state)[0]
    assert "vice city" in c.meta.title.lower() and c.meta.source == "template"


def test_cutscene_without_ai_reweights(cfg, state):
    a, v = _cutscene_synth(ai_ok=False).build()
    c = score_chunk(make_chunk(), a, v, cfg, state)[0]
    assert c.track == "cutscene"
    assert set(c.signals) == {"spectacle", "music_swell"}
    w = cfg.scoring.cutscene_weights
    expect = 100 * (w.spectacle * c.signals["spectacle"] + w.music_swell * c.signals["music_swell"]) / (
        w.spectacle + w.music_swell)
    assert abs(c.signal_score - expect) < 0.05
