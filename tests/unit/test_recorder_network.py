"""Real live YouTube recording (needs internet). Run with:  pytest -m network tests/unit/test_recorder_network.py"""
from __future__ import annotations

from pathlib import Path

import pytest

from clipper import ffmpeg
from clipper.config import load_config
from clipper.recorder import Recorder

LIVE_URL = "https://www.youtube.com/@LofiGirl/live"


@pytest.mark.network
def test_record_real_live_stream(state, request):
    if "network" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("network test: run with -m network")
    cfg = load_config(overrides={"recorder": {"chunk_seconds": 15, "poll_s": 1.0},
                                 "stream": {"live_poll_s": 30}})
    rec = Recorder(cfg, state, LIVE_URL, max_chunks=2)
    assert rec.run() == 0
    chunks = state.chunks(state.current_stream())
    assert [c.idx for c in chunks] == [0, 1]
    assert chunks[0].start_s == 0.0
    assert chunks[1].start_s == pytest.approx(chunks[0].end_s, abs=1e-3)
    for c in chunks:
        assert 10 <= c.duration_s <= 25
        info = ffmpeg.probe(c.path)
        kinds = {s["codec_type"] for s in info["streams"]}
        assert kinds == {"video", "audio"}
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert int(v["height"]) <= 720
        assert Path(c.path).stat().st_size > 100_000
