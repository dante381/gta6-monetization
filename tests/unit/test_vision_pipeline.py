"""Vision pipeline on the synthetic chunks: real GPU decode, fake Claude."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clipper import paths
from clipper.analysis import vision
from clipper.models import Chunk

pytestmark = pytest.mark.gpu


def _chunk(src, truth, idx=7):
    return Chunk(idx=idx, path=str(src), start_s=idx * truth["duration"], duration_s=truth["duration"],
                 stream_id="test-stream")


def _in(t, rng, margin):
    return rng[0] + margin <= t <= rng[1] - margin


def _out(t, rng, margin):
    return t < rng[0] - margin or t > rng[1] + margin


def _events():
    p = paths.DATA.logs / "clips.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_full_chunk_cv_grids_and_claude_merge(synthetic_chunk, cfg, claude_scenario, fake_claude_calls):
    src, truth = synthetic_chunk
    ev = truth["events"]
    claude_scenario({"grid": {"default": {"scene_tag": "test pattern", "action": 5}, "ranges": [
        {"from": ev["cutscene"][0], "to": ev["cutscene"][1], "kind": "cutscene", "title_text": "THE HEIST",
         "scene_tag": "bank vault", "action": 3},
        {"from": ev["gunfire"][0], "to": ev["gunfire"][1], "hud_events": ["wanted_up"], "wanted_stars": 2,
         "action": 9},
        {"from": ev["loading"][0], "to": ev["loading"][1], "kind": "loading", "action": 0},
    ]}})
    ticks = []
    ch = _chunk(src, truth)
    vf = vision.analyze_video(ch, cfg, on_tick=lambda: ticks.append(1))

    # counts: 300 s -> 100 frames every 3 s, 5 grids of 20, 600 motion samples at 2 fps
    assert vf.chunk_idx == 7 and vf.ai_ok
    assert len(vf.frames) == 100 and len(vf.frame_paths) == 100
    assert [f.t for f in vf.frames] == [i * 3.0 for i in range(100)]
    assert all(Path(p).is_file() for p in vf.frame_paths.values())
    assert Path(vf.frame_paths[0.0]).parent == paths.DATA.frames / "test-stream" / "7"
    assert len(vf.grid_paths) == 5 and all(Path(g).is_file() for g in vf.grid_paths)
    assert Path(vf.grid_paths[0]).parent == paths.DATA.grids / "test-stream" / "7"
    assert not list((paths.DATA.frames / "test-stream" / "7").glob("_*"))   # raw temp files removed
    m = vf.motion
    assert len(m.t) == 600 and m.t[1] == 0.5
    assert all(len(a) == 600 for a in (m.motion, m.scene_change, m.flash, m.shake, m.letterbox, m.static))

    # letterbox only in the cutscene window (frames and motion series)
    for f in vf.frames:
        if _in(f.t, ev["cutscene"], 2):
            assert f.letterbox, f.t
        elif _out(f.t, ev["cutscene"], 2):
            assert not f.letterbox, f.t
    in_cut = (m.t > ev["cutscene"][0] + 1) & (m.t < ev["cutscene"][1] - 1)
    out_cut = (m.t < ev["cutscene"][0] - 1) | (m.t > ev["cutscene"][1] + 1)
    assert m.letterbox[in_cut].all() and not m.letterbox[out_cut].any()

    # static in the loading window only
    in_load = (m.t > ev["loading"][0] + 2) & (m.t < ev["loading"][1] - 2)
    out_load = (m.t < ev["loading"][0] - 2) | (m.t > ev["loading"][1] + 2)
    assert m.static[in_load].all() and not m.static[out_load].any()

    # motion is high in the moving pattern, ~0 on the black loading screen
    moving = (m.t > 2) & (m.t < ev["gunfire"][0])
    assert m.motion[moving].mean() > 0.05
    assert m.motion[in_load].max() < 0.005
    assert m.motion[moving].mean() > 20 * max(1e-4, m.motion[in_load].mean())
    assert np.all((m.motion >= 0) & (m.motion <= 1)) and np.all((m.shake >= 0) & (m.shake <= 1))

    # Claude labels merged with the CV letterbox
    assert all(f.from_ai for f in vf.frames)
    by_t = {f.t: f for f in vf.frames}
    cut = [f for f in vf.frames if _in(f.t, ev["cutscene"], 2)]
    assert cut and all(f.kind == "cutscene" and f.title_text == "THE HEIST" and f.scene_tag == "bank vault"
                       and f.letterbox for f in cut)
    gun = [f for f in vf.frames if ev["gunfire"][0] <= f.t < ev["gunfire"][1]]
    assert gun and all(f.hud_events == ["wanted_up"] and f.wanted_stars == 2 and f.action == 9 for f in gun)
    assert by_t[0.0].kind == "gameplay" and by_t[0.0].scene_tag == "test pattern" and not by_t[0.0].letterbox

    # ONE claude call: prompt has TIMESTAMPS + every grid path, low effort, model set, grid dir allowed
    calls = fake_claude_calls()
    assert len(calls) == 1
    argv, prompt = calls[0]["argv"], calls[0]["prompt"]
    ts_line = next(l for l in prompt.splitlines() if l.startswith("TIMESTAMPS:"))
    assert ts_line == "TIMESTAMPS: " + ",".join(str(i * 3) for i in range(100))
    for g in vf.grid_paths:
        assert str(Path(g).resolve()) in prompt
    assert argv[argv.index("--effort") + 1] == "low" == cfg.claude.effort_grid
    assert argv[argv.index("--model") + 1] == cfg.claude.model
    assert str(Path(vf.grid_paths[0]).resolve().parent) in argv
    assert '"frames"' in argv[argv.index("--json-schema") + 1]
    assert ticks, "on_tick never called"


@pytest.mark.parametrize("mode", ["limit", "garbage", "crash"])
def test_claude_failure_falls_back_to_cv(synthetic_short, cfg, monkeypatch, mode, fake_claude_calls):
    src, truth = synthetic_short
    ev = truth["events"]
    monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
    vf = vision.analyze_video(_chunk(src, truth, idx=3), cfg)
    assert len(fake_claude_calls()) == 1
    assert vf.ai_ok is False
    assert len(vf.frames) == 20 and len(vf.grid_paths) == 1
    assert not any(f.from_ai for f in vf.frames)
    for f in vf.frames:
        if _in(f.t, ev["cutscene"], 1):
            assert f.kind == "cutscene" and f.letterbox, f.t
        elif _in(f.t, ev["loading"], 0):
            assert f.kind == "loading" and f.action == 0, f.t
        elif _out(f.t, ev["cutscene"], 1) and _out(f.t, ev["loading"], 1):
            assert f.kind == "gameplay" and not f.letterbox, f.t
    game_actions = [f.action for f in vf.frames if f.kind == "gameplay"]
    assert game_actions and max(game_actions) >= 1
    unav = [e for e in _events() if e["kind"] == "claude_unavailable"]
    assert len(unav) == 1 and unav[0]["chunk_idx"] == 3 and unav[0]["stage"] == "grid"


def test_use_ai_false_skips_claude(synthetic_short, cfg, fake_claude_calls):
    src, truth = synthetic_short
    vf = vision.analyze_video(_chunk(src, truth), cfg, use_ai=False)
    assert fake_claude_calls() == []
    assert vf.ai_ok is False and len(vf.frames) == 20 and not any(f.from_ai for f in vf.frames)


def test_rerun_replaces_old_outputs(synthetic_short, cfg):
    src, truth = synthetic_short
    ch = _chunk(src, truth)
    fdir = vision.frames_dir(ch)
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / "t9999.0.jpg").write_bytes(b"stale")
    vf = vision.analyze_video(ch, cfg, use_ai=False)
    assert not (fdir / "t9999.0.jpg").exists()
    assert len(list(fdir.glob("*.jpg"))) == len(vf.frame_paths) == 20
