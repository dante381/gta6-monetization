"""Recorder end to end with real ffmpeg (copy only) on a local source read in real time (-re).

yt-dlp is faked: "resolve" hands back local file paths (or a local HTTP URL that stalls).
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from clipper import ffmpeg, heartbeat, paths
from clipper.config import load_config
from clipper.recorder import Recorder
from clipper.state import StateDB

FIX = paths.TMP / "test_fixtures"
VID = "vid123"
CHUNK_S = 2.0


def make_source(seconds: int) -> Path:
    """Small H.264/AAC TS with a keyframe every 0.48 s (CPU encode is fine for fixtures)."""
    FIX.mkdir(parents=True, exist_ok=True)
    out = FIX / f"recorder_src_{seconds}s.ts"
    if not out.exists():
        tmp = out.with_suffix(".part.ts")
        ffmpeg.run(["-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=25:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "12", "-keyint_min", "12",
                    "-sc_threshold", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
                    "-f", "mpegts", str(tmp)], timeout=120)
        tmp.replace(out)
    return out


@pytest.fixture(scope="module")
def src20():
    return make_source(20)


@pytest.fixture(scope="module")
def src5():
    return make_source(5)


class FakeYtdlp:
    """statuses: per call (all calls); urls: per URL-resolving call. Last entry repeats."""

    def __init__(self, urls: list[list[str]], statuses: list[str] | None = None, after: str = "is_live"):
        self.urls = urls
        self.statuses = statuses or []
        self.after = after
        self.calls: list[list[str]] = []
        self.url_calls = 0

    def __call__(self, cmd, *, timeout, on_tick=None):
        n = len(self.calls)
        self.calls.append(cmd)
        status = self.statuses[n] if n < len(self.statuses) else self.after
        out = json.dumps({"id": VID, "title": "test stream", "live_status": status,
                          "is_live": status == "is_live"}) + "\n"
        if "urls" in cmd:
            if status == "is_live":
                out += "\n".join(self.urls[min(self.url_calls, len(self.urls) - 1)]) + "\n"
            self.url_calls += 1
        return 0, out, ""


def rcfg(**rec):
    r = {"chunk_seconds": CHUNK_S, "stall_seconds": 10, "reconnect_backoff_s": [0.2], "end_confirm_checks": 2,
         "end_check_interval_s": 0.1, "poll_s": 0.2, "input_args": ["-re"]}
    r.update(rec)
    return load_config(overrides={"recorder": r, "stream": {"live_poll_s": 0.1}})


def run_bg(rec: Recorder, timeout: float = 45, watch=None):
    """Run rec.run() in a thread; `watch()` is called every 0.1 s from the test thread."""
    res: dict = {}

    def target():
        try:
            res["rc"] = rec.run()
        except BaseException as e:  # surface in the test
            res["exc"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    end = time.monotonic() + timeout
    while t.is_alive() and time.monotonic() < end:
        if watch:
            watch()
        t.join(0.1)
    if t.is_alive():
        if rec.proc:
            ffmpeg.kill_tree(rec.proc)
        pytest.fail("recorder did not finish in time")
    if "exc" in res:
        raise res["exc"]
    return res["rc"]


def events(kind: str) -> list[dict]:
    p = paths.DATA.logs / "clips.jsonl"
    if not p.exists():
        return []
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in recs if r["kind"] == kind]


def assert_contiguous(chunks, first_idx=0, first_start=0.0):
    assert [c.idx for c in chunks] == list(range(first_idx, first_idx + len(chunks)))
    t = first_start
    for c in chunks:
        assert c.start_s == pytest.approx(t, abs=2e-3)
        assert c.duration_s > 0
        assert Path(c.path).exists()
        t = c.start_s + c.duration_s


def test_chunks_registered_with_cumulative_start(state, src20):
    """Two inputs (video + audio playlists like live YouTube) -> closed chunks with cumulative start_s."""
    fake = FakeYtdlp([[str(src20), str(src20)]])
    rec = Recorder(rcfg(), state, "https://youtube.example/live", runner=fake, max_chunks=3)
    assert run_bg(rec) == 0
    sid = state.current_stream()
    assert sid.startswith(f"{VID}-") and len(sid) == len(VID) + 9
    chunks = state.chunks(sid)
    assert len(chunks) == 3
    assert_contiguous(chunks)
    for c in chunks:
        assert 1.5 <= c.duration_s <= 3.5
        assert Path(c.path).parent == paths.DATA.chunks / sid
        assert Path(c.path).name == f"chunk_{c.idx:05d}.ts"
    info = ffmpeg.probe(chunks[0].path)
    kinds = sorted(s["codec_type"] for s in info["streams"])
    assert kinds == ["audio", "video"]
    assert len(events("chunk_recorded")) == 3
    assert not list((paths.DATA.chunks / sid).glob("chunk_00003.ts"))  # partial discarded at max
    assert heartbeat.read("recorder") is not None
    assert state.kv_get("recorder.done") is False


def test_reconnect_after_ffmpeg_killed_continues_numbering(state, src20):
    fake = FakeYtdlp([[str(src20)]])
    rec = Recorder(rcfg(), state, "u", runner=fake, max_chunks=4)
    killed = {"done": False}
    db = StateDB()

    def watch():
        sid = db.current_stream()
        if not killed["done"] and sid and len(db.chunks(sid)) >= 2 and rec.proc is not None:
            time.sleep(0.6)  # let the next segment get some media
            ffmpeg.kill_tree(rec.proc)
            killed["done"] = True

    try:
        assert run_bg(rec, watch=watch) == 0
    finally:
        db.close()
    assert killed["done"]
    chunks = state.chunks(state.current_stream())
    assert len(chunks) >= 4
    assert_contiguous(chunks)
    rc = events("reconnect")
    assert len(rc) == 1 and rc[0]["reason"] in ("exit", "eof")
    assert rc[0]["next_idx"] in (2, 3)    # 3 if the killed partial had >= 1 s of media
    assert fake.url_calls >= 2       # URL re-resolved for the new session
    assert rec.sessions == 2


def test_stream_end_sets_done(state, src5):
    # wait_until_live, first probe -> live; after the 5 s source ends: not live x2 -> end
    fake = FakeYtdlp([[str(src5)]], statuses=["is_live", "is_live"], after="was_live")
    rec = Recorder(rcfg(), state, "u", runner=fake)
    assert run_bg(rec) == 0
    assert state.kv_get("recorder.done") is True
    chunks = state.chunks(state.current_stream())
    assert_contiguous(chunks)
    assert sum(c.duration_s for c in chunks) == pytest.approx(5.0, abs=0.6)
    assert chunks[-1].duration_s < CHUNK_S + 0.6     # last partial chunk registered
    ev = events("stream_end")
    assert len(ev) == 1 and ev[0]["chunks"] == len(chunks)
    assert len(fake.calls) == 4      # wait, probe, probe (not live), confirm check
    assert not events("reconnect")


def test_resume_continues_index_and_start(state, src20, src5):
    state.set_current_stream(f"{VID}-20200101")       # earlier run of the same video (other day)
    fake = FakeYtdlp([[str(src20)]])
    assert run_bg(Recorder(rcfg(), state, "u", runner=fake, max_chunks=2)) == 0
    sid = state.current_stream()
    assert sid == f"{VID}-20200101"
    first = state.chunks(sid)
    assert [c.idx for c in first] == [0, 1]
    # simulate a crash that left an unregistered partial chunk on disk
    shutil.copy(src5, paths.DATA.chunks / sid / "chunk_00002.ts")
    assert run_bg(Recorder(rcfg(), state, "u", runner=FakeYtdlp([[str(src20)]]), max_chunks=2)) == 0
    assert state.current_stream() == sid
    chunks = state.chunks(sid)
    assert [c.idx for c in chunks] == [0, 1, 2, 3, 4]
    assert_contiguous(chunks)
    assert chunks[2].start_s == pytest.approx(first[1].end_s, abs=2e-3)
    assert chunks[2].duration_s == pytest.approx(5.0, abs=0.5)    # recovered crash leftover


class _StallHandler(BaseHTTPRequestHandler):
    data = b""
    release: threading.Event

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "video/mp2t")
        self.end_headers()
        try:
            self.wfile.write(self.data)
            self.wfile.flush()
            self.release.wait(30)   # connection stays open, no more bytes: a stalled live source
        except OSError:
            pass

    def log_message(self, *a):
        pass


def test_stall_detected_then_reconnect_then_end(state, src20, src5):
    blob = src20.read_bytes()
    _StallHandler.data = blob[: len(blob) * 5 // 20]       # ~5 s of media, then silence
    _StallHandler.release = threading.Event()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StallHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/live.ts"
    try:
        fake = FakeYtdlp([[url], [str(src5)]], statuses=["is_live"] * 3, after="not_live")
        cfg = rcfg(stall_seconds=2, input_args=["-re", "-analyzeduration", "1000000"])
        rec = Recorder(cfg, state, "u", runner=fake)
        assert run_bg(rec) == 0
    finally:
        _StallHandler.release.set()
        srv.shutdown()
        srv.server_close()
    rc = events("reconnect")
    assert [r["reason"] for r in rc] == ["stall"]
    chunks = state.chunks(state.current_stream())
    assert_contiguous(chunks)
    assert sum(c.duration_s for c in chunks) == pytest.approx(10.0, abs=2.5)
    assert state.kv_get("recorder.done") is True
