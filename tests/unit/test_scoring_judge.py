"""Judge gate, Claude call details, failure fallback, final score and routing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper import paths
from clipper.models import Candidate
from clipper.scoring import final_score, route_for, score_chunk
from clipper.scoring import judge as judge_mod
from tests.unit.test_scoring_synth import Synth, make_chunk


def _hot_synth(duration=300.0, t0=75.0, t1=85.0):
    s = Synth(duration).gunfire(t0, t1)
    s.set_frames(t0, duration, wanted_stars=5)
    s.set_frames(t0, t0 + 3, hud_events=["wanted_up"])
    return s


@pytest.fixture
def fake_grab(monkeypatch):
    """Replace the GPU frame grab with dummy JPEG files (the real one is tested under @gpu)."""
    calls = []

    def grab(src, times, out_dir, width=1280, prefix="still", on_tick=None):
        calls.append({"src": str(src), "times": list(times), "out_dir": str(out_dir), "width": width})
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ps = []
        for i, t in enumerate(times):
            p = out / f"{prefix}_{i:02d}_{t:.1f}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xd9")
            ps.append(p)
        return ps

    monkeypatch.setattr(judge_mod.ffmpeg, "grab_frames", grab)
    return calls


def test_final_score_formula():
    assert final_score(70, 8, 0.3) == pytest.approx(0.7 * 70 + 0.3 * 80)
    assert final_score(70, None, 0.3) == 70
    assert final_score(100, 10, 0.3) == 100
    assert final_score(60, 1, 0.5) == pytest.approx(35)


def test_routing_thresholds(cfg):
    th = cfg.locked.thresholds
    assert route_for(th.backup - 0.01, cfg) == "drop"
    assert route_for(th.backup, cfg) == "backup"
    # the legacy publish line decides nothing: publishing is chosen later by clipper.selection
    assert route_for(th.publish - 0.01, cfg) == "backup"
    assert route_for(th.publish, cfg) == "backup"
    assert route_for(100, cfg) == "backup"


def test_judge_gate_calls_claude_with_frames(cfg, state, fake_grab, fake_claude_calls, claude_scenario):
    claude_scenario({"judge": {"rating": 9, "title": "Five star rampage", "hook": "5 stars in 10 seconds"}})
    a, v = _hot_synth().build()
    chunk = make_chunk(path="E:/fake/chunk_00000.ts")
    cands = score_chunk(chunk, a, v, cfg, state)
    assert len(cands) == 1
    c = cands[0]
    assert c.signal_score >= cfg.locked.thresholds.judge_gate

    # frames: judge_frames full-res stills spread across the window, chunk-relative times
    assert len(fake_grab) == 1
    g = fake_grab[0]
    assert g["src"] == chunk.path
    assert len(g["times"]) == cfg.scoring.judge_frames
    assert all(c.start_s - chunk.start_s <= t < c.end_s - chunk.start_s for t in g["times"])
    assert Path(g["out_dir"]) == paths.DATA.work / "judge" / c.id

    calls = fake_claude_calls()
    assert len(calls) == 1
    argv, prompt = calls[0]["argv"], calls[0]["prompt"]
    assert argv[argv.index("--effort") + 1] == "medium"
    assert argv[argv.index("--model") + 1] == cfg.claude.model
    assert Path(argv[argv.index("--add-dir") + 1]) == (paths.DATA.work / "judge" / c.id).resolve()
    assert "rating" in json.loads(argv[argv.index("--json-schema") + 1])["properties"]
    for p in Path(g["out_dir"]).glob("*.jpg"):
        assert str(p.resolve()) in prompt
    assert "GTA 6" in prompt and "gameplay" in prompt and "city street" in prompt
    assert "wanted level up to 5 stars" in prompt
    assert "$" not in prompt.replace("$game", "")          # every placeholder substituted

    assert c.judge["rating"] == 9
    assert c.final_score == pytest.approx(round(0.7 * c.signal_score + 27, 2), abs=0.01)
    assert c.route == route_for(c.final_score, cfg)
    assert c.meta.source == "judge"
    assert "GTA 6" in c.meta.title and c.meta.hook == "5 stars in 10 seconds"


def test_below_gate_no_judge(cfg, state, fake_grab, fake_claude_calls):
    a, v = Synth(300).bang(100).build()
    c = score_chunk(make_chunk(path="E:/fake/c.ts"), a, v, cfg, state)[0]
    assert c.signal_score < cfg.locked.thresholds.judge_gate
    assert fake_grab == [] and fake_claude_calls() == []
    assert c.judge is None and c.final_score == c.signal_score
    assert c.meta.source == "template"


def test_judge_failure_falls_back_to_signal(cfg, state, fake_grab, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "limit")
    a, v = _hot_synth().gunfire(200, 210).build()
    cands = score_chunk(make_chunk(path="E:/fake/c.ts"), a, v, cfg, state)
    gated = [c for c in cands if c.signal_score >= cfg.locked.thresholds.judge_gate]
    assert gated
    for c in cands:
        assert c.judge is None
        assert c.final_score == c.signal_score
        assert c.meta.source == "template" and "GTA 6" in c.meta.title
    ev = [json.loads(l) for l in (paths.DATA.logs / "clips.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for e in ev if e["kind"] == "claude_unavailable") == 1    # stops after the first failure


def test_judge_garbage_output_falls_back(cfg, state, fake_grab, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "garbage")
    a, v = _hot_synth().build()
    c = score_chunk(make_chunk(path="E:/fake/c.ts"), a, v, cfg, state)[0]
    assert c.judge is None and c.final_score == c.signal_score


def test_low_rating_pulls_final_down(cfg, state, fake_grab, claude_scenario):
    claude_scenario({"judge": {"rating": 2}})
    a, v = _hot_synth().build()
    c = score_chunk(make_chunk(path="E:/fake/c.ts"), a, v, cfg, state)[0]
    assert c.final_score == pytest.approx(0.7 * c.signal_score + 6, abs=0.01)
    assert c.final_score < c.signal_score


def test_judge_times_skip_outside_chunk():
    from clipper.models import Chunk
    chunk = Chunk(idx=1, path="x", start_s=300, duration_s=300)
    c = Candidate(id="x", chunk_idx=1, start_s=290, peak_s=296, end_s=330, track="gameplay")
    ts = judge_mod.judge_times(c, chunk, 300, 4)
    assert ts and all(0 <= t < 300 for t in ts)
    assert len(ts) == 3                       # the first quarter is in the previous chunk
    c2 = Candidate(id="y", chunk_idx=1, start_s=580, peak_s=590, end_s=620, track="gameplay")
    assert all(t < 300 for t in judge_mod.judge_times(c2, chunk, 300, 4))


@pytest.mark.gpu
def test_judge_real_frames_on_synthetic_short(cfg, state, synthetic_short, fake_claude_calls):
    path, truth = synthetic_short
    g0, g1 = truth["events"]["gunfire"]
    s = _hot_synth(60.0, g0, g1)
    a, v = s.build()
    chunk = make_chunk(path=str(path), duration=60.0)
    ticks = []
    cands = score_chunk(chunk, a, v, cfg, state, on_tick=lambda: ticks.append(1))
    c = next(c for c in cands if c.judge)
    d = paths.DATA.work / "judge" / c.id
    jpgs = sorted(d.glob("*.jpg"))
    assert 1 <= len(jpgs) <= cfg.scoring.judge_frames
    assert all(p.stat().st_size > 1000 for p in jpgs)
    prompt = fake_claude_calls()[0]["prompt"]
    assert all(str(p.resolve()) in prompt for p in jpgs)
    assert c.final_score == pytest.approx(round(0.7 * c.signal_score + 0.3 * c.judge["rating"] * 10, 2), abs=0.01)
