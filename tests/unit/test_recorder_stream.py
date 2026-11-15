"""clipper.recorder.stream with a faked yt-dlp runner (no network)."""
from __future__ import annotations

import json

import pytest

from clipper import paths
from clipper.recorder import stream


class Runner:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, *, timeout, on_tick=None):
        self.calls.append(cmd)
        r = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return r


def info_line(status="is_live", vid="abc123", title="GTA 6 stream"):
    return json.dumps({"id": vid, "title": title, "live_status": status, "is_live": status == "is_live"})


def test_live_status_parses_info_and_command(cfg):
    r = Runner((0, info_line() + "\n", ""))
    st = stream.live_status("https://www.youtube.com/@x/live", cfg, runner=r)
    assert st == {"status": "is_live", "video_id": "abc123", "title": "GTA 6 stream"}
    cmd = r.calls[0]
    assert cmd[0] == str(paths.VENV / "Scripts" / "yt-dlp.exe")
    assert cmd[cmd.index("--cache-dir") + 1] == str(paths.CACHE / "yt-dlp")
    assert "--no-warnings" in cmd
    assert cmd[cmd.index("-f") + 1] == cfg.stream.format
    assert "urls" not in cmd
    assert cmd[-1] == "https://www.youtube.com/@x/live"


def test_resolve_hls_two_urls(cfg):
    out = info_line() + "\nhttps://v.example/video.m3u8\nhttps://a.example/audio.m3u8\n"
    r = Runner((0, out, ""))
    urls = stream.resolve_hls("u", cfg, runner=r)
    assert urls == ["https://v.example/video.m3u8", "https://a.example/audio.m3u8"]
    assert "urls" in r.calls[0]


def test_resolve_hls_single_url(cfg):
    r = Runner((0, info_line() + "\nhttps://x.example/muxed.m3u8\n", ""))
    assert stream.resolve_hls("u", cfg, runner=r) == ["https://x.example/muxed.m3u8"]


def test_resolve_hls_raises_when_no_urls(cfg):
    r = Runner((0, info_line("was_live") + "\nNA\n", ""))
    with pytest.raises(stream.StreamError):
        stream.resolve_hls("u", cfg, runner=r)


@pytest.mark.parametrize("stderr,expected", [
    ("ERROR: [youtube:tab] @x: The channel is not currently live", "not_live"),
    ("ERROR: [youtube] abc: This live event will begin in 3 hours.", "is_upcoming"),
    ("ERROR: unable to download webpage: timed out", "error"),
])
def test_live_status_errors(cfg, stderr, expected):
    st = stream.live_status("u", cfg, runner=Runner((1, "", stderr)))
    assert st["status"] == expected
    assert st["video_id"] is None


def test_runner_exception_is_error(cfg):
    def boom(cmd, *, timeout, on_tick=None):
        raise FileNotFoundError("yt-dlp.exe")
    assert stream.live_status("u", cfg, runner=boom)["status"] == "error"


def test_status_falls_back_to_is_live_flag(cfg):
    out = json.dumps({"id": "v", "title": "t", "live_status": None, "is_live": True})
    assert stream.live_status("u", cfg, runner=Runner((0, out, "")))["status"] == "is_live"


class FakeHB:
    def __init__(self):
        self.ticks = 0

    def tick(self, **_):
        self.ticks += 1


def test_wait_until_live_polls_and_ticks(cfg):
    cfg.stream.live_poll_s = 12
    r = Runner((1, "", "The channel is not currently live"), (0, info_line("is_upcoming"), ""),
               (0, info_line("is_live", vid="live1"), ""))
    slept: list[float] = []
    hb = FakeHB()
    st = stream.wait_until_live("u", cfg, hb, runner=r, sleep=slept.append)
    assert st["status"] == "is_live" and st["video_id"] == "live1"
    assert len(r.calls) == 3
    assert max(slept) <= 5.0                      # never sleeps more than 5 s between ticks
    assert sum(slept) == pytest.approx(24, abs=0.5)
    assert hb.ticks >= 6
