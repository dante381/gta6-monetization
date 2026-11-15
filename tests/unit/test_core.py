"""Smoke tests for the frozen core contracts (config, state, heartbeat, claude wrapper, ffmpeg wrapper)."""
from __future__ import annotations

import time

import pytest

from clipper import claude_cli, heartbeat
from clipper.models import Candidate, Chunk, ClipMeta, QueueItem


def test_config_loads(cfg):
    assert cfg.locked.thresholds.publish == 75
    assert cfg.locked.caps.daily_total == 10
    assert sum(cfg.scoring.gameplay_weights.values()) == 100
    assert sum(cfg.scoring.cutscene_weights.values()) == 100
    assert cfg.claude.model == "claude-sonnet-5-5"


def test_config_overrides():
    from clipper.config import load_config
    c = load_config(overrides={"locked": {"test_mode": False}, "publish": {"slot_hours": 3}})
    assert c.locked.test_mode is False and c.publish.slot_hours == 3
    assert c.locked.thresholds.publish == 75


def test_state_chunks_and_queue(state):
    sid = state.current_stream()
    for i in range(3):
        state.add_chunk(Chunk(idx=i, path=f"c{i}.ts", start_s=i * 300.0, duration_s=300.0, stream_id=sid))
    assert state.last_chunk_idx(sid) == 2
    assert state.next_chunk_to_process(sid).idx == 0
    state.update_chunk(sid, 0, status="processed")
    assert state.next_chunk_to_process(sid).idx == 1
    cov = state.chunks_covering(sid, 290, 310)
    assert [c.idx for c in cov] == [0, 1]

    c = Candidate(id="x1", chunk_idx=0, start_s=10, peak_s=16, end_s=40, track="gameplay",
                  final_score=80, route="publish", scene_tag="beach", meta=ClipMeta(title="t"))
    state.save_candidate(c, sid)
    back = state.candidates(sid)[0]
    assert back.meta.title == "t" and back.route == "publish"
    assert state.selected_near(sid, 30, 20)[0].id == "x1"
    assert "beach" in state.selected_scene_tags(sid)

    state.queue_add(QueueItem(clip_id="x1", clip_path="a.mp4", sidecar_path="a.json", final_score=80,
                              track="gameplay", kind="publish"))
    state.queue_update("x1", status="uploaded", video_id="vid")
    assert state.queue_get("x1").video_id == "vid"
    state.record_upload("x1", "2026-11-15", "publish", "vid")
    assert state.uploads_on("2026-11-15") == 1

    state.set_publish_hold(True, "repair")
    assert state.publish_hold()


def test_heartbeat():
    hb = heartbeat.Heartbeat("unit", interval=0)
    hb.tick(stage="x")
    a = heartbeat.age("unit")
    assert a is not None and a < 5
    assert heartbeat.read("unit")["stage"] == "x"


def test_claude_fake_grid(fake_claude_calls):
    prompt = "Label tiles.\nTIMESTAMPS: 0,3,6\n"
    out = claude_cli.call(prompt, schema="grid", effort="low")
    assert [f["t"] for f in out["frames"]] == [0, 3, 6]
    argv = fake_claude_calls()[0]["argv"]
    assert "--model" in argv and argv[argv.index("--model") + 1] == "claude-sonnet-5-5"
    assert "--no-session-persistence" in argv and "--safe-mode" in argv
    assert argv[argv.index("--effort") + 1] == "low"


def test_claude_fake_limit(monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "limit")
    with pytest.raises(claude_cli.ClaudeUnavailable):
        claude_cli.call("x", schema="judge", effort="medium")


def test_claude_fake_garbage(monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "garbage")
    with pytest.raises(claude_cli.ClaudeError):
        claude_cli.call("x", schema="judge", effort="medium")


@pytest.mark.gpu
def test_ffmpeg_nvenc_smoke(tmp_path):
    from clipper import ffmpeg
    out = tmp_path / "n.mp4"
    ffmpeg.run_gpu(["-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=1", "-c:v", "h264_nvenc", str(out)],
                   max_wait_s=0)
    assert ffmpeg.duration(out) > 0.5


def test_ffmpeg_error_classification():
    from clipper import ffmpeg
    assert ffmpeg.is_gpu_error("OpenEncodeSessionEx failed: out of memory")
    with pytest.raises(ffmpeg.FfmpegError):
        ffmpeg.run(["-i", "does_not_exist.ts", "-f", "null", "-"])


def test_synthetic_fixture(synthetic_short):
    from clipper import ffmpeg
    path, truth = synthetic_short
    assert abs(ffmpeg.duration(path) - 60) < 1.5
    assert truth["events"]["gunfire"] == [15.0, 17.0]
