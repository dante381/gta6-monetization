"""score_chunk persistence: state rows, seen scenes, event log, ordering, stream seconds."""
from __future__ import annotations

import json

from clipper import paths
from clipper.scoring import score_chunk
from tests.unit.test_scoring_synth import Synth, make_chunk


def _events():
    p = paths.DATA.logs / "clips.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def test_all_candidates_saved_sorted_and_logged(cfg, state):
    s = Synth(300).bang(40).gunfire(150, 160)
    s.set_frames(150, 300, scene_tag="docks")
    a, v = s.build()
    cands = score_chunk(make_chunk(idx=2, start_s=600), a, v, cfg, state)
    assert len(cands) == 2
    assert [c.peak_s for c in cands] == sorted(c.peak_s for c in cands)
    weak, strong = cands
    assert weak.route == "drop" and strong.route != "drop"
    assert 600 + 40 <= weak.peak_s < 600 + 41                     # stream seconds
    saved = {c.id: c for c in state.candidates("test-stream")}
    assert set(saved) == {weak.id, strong.id}                      # drops saved too (calibration)
    assert saved[strong.id].final_score == strong.final_score
    assert saved[strong.id].meta.title == strong.meta.title
    # seen scenes: only the selected one
    assert state.scene_count("docks") == 1
    assert state.scene_count("city street") == 0
    ev = [e for e in _events() if e["kind"] == "candidate"]
    assert {e["id"] for e in ev} == {weak.id, strong.id}
    e = next(e for e in ev if e["id"] == strong.id)
    assert e["route"] == strong.route and e["track"] == "gameplay" and e["title"] == strong.meta.title


def test_rescoring_same_chunk_is_not_a_repeat_of_itself(cfg, state):
    a, v = Synth(300).gunfire(75, 85).build()
    first = score_chunk(make_chunk(), a, v, cfg, state)[0]
    again = score_chunk(make_chunk(), a, v, cfg, state)[0]
    assert first.id == again.id
    assert "repeat" not in again.penalties
    assert again.signal_score == first.signal_score


def test_on_tick_called(cfg, state):
    ticks = []
    a, v = Synth(300).gunfire(75, 85).build()
    score_chunk(make_chunk(), a, v, cfg, state, on_tick=lambda: ticks.append(1))
    assert ticks


def test_zero_duration_chunk_uses_feature_length(cfg, state):
    a, v = Synth(120).gunfire(50, 60).build()
    c = score_chunk(make_chunk(duration=0.0), a, v, cfg, state)[0]
    assert 50 <= c.peak_s < 60


def test_candidate_round_trip(cfg, state):
    a, v = Synth(300).gunfire(75, 85).radio_on(80, 140).build()
    c = score_chunk(make_chunk(), a, v, cfg, state)[0]
    back = state.candidates("test-stream")[0]
    assert json.loads(json.dumps(back.to_dict())) == json.loads(json.dumps(c.to_dict()))
