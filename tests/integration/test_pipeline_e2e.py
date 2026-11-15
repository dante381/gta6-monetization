"""End to end on the synthetic 300 s chunk: REAL audio, vision (GPU), scoring, production (GPU),
queue and publish tick (test mode); only Claude is faked.

Timeline (see tests/fixtures/make_synthetic.py): gunfire 75-85, radio 100-160, cutscene 170-230,
loading 240-270, explosion 285-288.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from clipper import ffmpeg, paths
from clipper.config import load_config
from clipper.models import Chunk
from clipper.state import StateDB
from clipper.worker import Worker

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

SCENARIO = {
    "grid": {
        "default": {"kind": "gameplay", "action": 3, "scene_tag": "city street"},
        "ranges": [
            {"from": 72, "to": 90, "kind": "gameplay", "action": 9, "hud_events": ["wanted_up"],
             "wanted_stars": 4, "scene_tag": "police chase"},
            {"from": 168, "to": 232, "kind": "cutscene", "action": 5, "title_text": "FIRST HEIST",
             "scene_tag": "vice city skyline"},
            {"from": 238, "to": 272, "kind": "loading", "action": 0, "scene_tag": None},
        ],
    },
    "judge": {"rating": 9, "title": "Wild 4-star police chase in GTA 6", "hook": "4 stars in 20 seconds"},
}


def _events(kind):
    p = paths.DATA.logs / "clips.jsonl"
    if not p.exists():
        return []
    return [r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
            if r["kind"] == kind]


def test_synthetic_chunk_to_clips(synthetic_chunk, claude_scenario, tmp_path, monkeypatch):
    src, truth = synthetic_chunk
    claude_scenario(SCENARIO)
    music = tmp_path / "music"
    music.mkdir()
    ffmpeg.run(["-f", "lavfi", "-i", "sine=f=330:d=30", str(music / "bed.wav")])
    monkeypatch.setattr(paths, "MUSIC", music)

    cfg = load_config(overrides={"locked": {"test_mode": True}})
    state = StateDB()
    sid = "e2e-stream"
    state.set_current_stream(sid)
    cdir = paths.DATA.chunks / sid
    cdir.mkdir(parents=True, exist_ok=True)
    dst = cdir / "chunk_00000.ts"
    shutil.copy2(src, dst)
    state.add_chunk(Chunk(idx=0, path=str(dst), start_s=0.0, duration_s=ffmpeg.duration(dst), stream_id=sid))
    state.kv_set("recorder.done", True)

    w = Worker(cfg, state, idle_sleep_s=0)
    assert w.run(max_loops=30) == 0

    cands = state.candidates(sid)
    assert cands, "no candidates scored"
    selected = [c for c in cands if c.route != "drop"]
    assert selected, f"nothing selected: {[(c.peak_s, c.track, round(c.final_score)) for c in cands]}"

    # the gunfire moment must be a gameplay candidate, the loading stretch must not be selected
    gun = [c for c in selected if 70 <= c.peak_s <= 95]
    assert gun and gun[0].track == "gameplay"
    assert not [c for c in selected if 245 <= c.peak_s <= 268]

    # clips exist, vertical, with sidecars
    clips = list(paths.DATA.ready.glob("*.mp4")) + list(paths.DATA.backup.glob("*.mp4"))
    assert clips
    for mp4 in clips:
        info = ffmpeg.probe(mp4)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert (v["width"], v["height"]) == (1080, 1920) and v["codec_name"] == "h264"
        side = json.loads(mp4.with_suffix(".json").read_text(encoding="utf-8"))
        assert "GTA 6" in side["title"] and "#Shorts" in side["hashtags"]
        assert 5 <= side["duration_s"] <= cfg.scoring.clip_max_s + 0.5

    # every kept clip is produced as a backup first; promotion to ready needs the percentile cut
    # (a single synthetic chunk has fewer samples than min_samples, the final decision logs its outcome)
    assert _events("selection")
    assert all(c.route in ("backup", "publish") for c in selected)
    assert len(state.history_scores(200)) == len(cands)       # every scored candidate, drops included
    assert not list(paths.DATA.ready.glob("*.mp4")) or _events("promoted")

    # test mode: nothing uploaded, would_upload logged for queued publish clips
    assert not _events("upload")
    pubs = [q for q in state.queue_items() if q.kind == "publish"]
    if pubs:
        assert _events("would_upload")
    # backups are never uploaded (publish.upload_backups false)
    assert not _events("would_upload") or all(e.get("item_kind") == "publish" for e in _events("would_upload"))

    # chunk footage cleaned up at stream end
    assert state.get_chunk(sid, 0).status == "deleted" and not dst.exists()
    assert _events("chunk_processed")


def test_claude_down_still_produces_clips(synthetic_chunk, monkeypatch, tmp_path):
    """Usage limit hit: scoring falls back to audio + motion, pipeline still finishes."""
    src, _ = synthetic_chunk
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "limit")
    cfg = load_config(overrides={"locked": {"test_mode": True}})
    state = StateDB()
    sid = "e2e-nolimit"
    state.set_current_stream(sid)
    cdir = paths.DATA.chunks / sid
    cdir.mkdir(parents=True, exist_ok=True)
    dst = cdir / "chunk_00000.ts"
    shutil.copy2(src, dst)
    state.add_chunk(Chunk(idx=0, path=str(dst), start_s=0.0, duration_s=ffmpeg.duration(dst), stream_id=sid))
    state.kv_set("recorder.done", True)
    assert Worker(cfg, state, idle_sleep_s=0).run(max_loops=30) == 0
    assert _events("claude_unavailable")
    assert state.candidates(sid), "no candidates without Claude"
    assert any(c.route != "drop" for c in state.candidates(sid))
