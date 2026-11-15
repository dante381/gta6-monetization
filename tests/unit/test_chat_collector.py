"""Chat collector: aggregation, privacy, reconnect, shutdown. No network (fake sources)."""
from __future__ import annotations

import logging
import time

import pytest

from clipper import heartbeat, paths
from clipper.chat.collector import Collector
from clipper.config import load_config

AUTHORS = ["AliceSecretName", "UCbobChannelId123", "CarolTheViewer"]


@pytest.fixture
def cfg():
    return load_config(overrides={"chat": {"collector_poll_s": 0.1, "bucket_s": 5}})


def mk(cfg, state, factory=None, **kw):
    return Collector(cfg, state, "vid1", "s1", source_factory=factory, **kw)


def rows(state):
    return state.chat_between("s1", 0, 10**12)


def test_bucket_floor_and_aggregation(cfg, state):
    c = mk(cfg, state)
    base = 1_700_000_000 // 5 * 5
    c.add(base + 0.2, "a", "OMG insane")
    c.add(base + 4.9, "b", "clip that")
    c.add(base + 5.0, "a", "hello")
    c.flush(now=base + 6)
    r = {x["t"]: x for x in rows(state)}
    assert set(r) == {base, base + 5}
    assert r[base]["msgs"] == 2 and r[base]["chatters"] == 2 and r[base]["clip_reqs"] == 1
    assert r[base]["hype"] > 0 and r[base + 5]["chatters"] == 1
    assert r[base]["t"] % 5 == 0


def test_incremental_flush_does_not_double_count(cfg, state):
    c = mk(cfg, state)
    base = 1_700_000_000 // 5 * 5
    c.add(base + 1, "a", "one")
    c.flush(now=base + 2)
    c.add(base + 2, "b", "two")
    c.add(base + 3, "a", "three")
    c.flush(now=base + 4)
    c.flush(now=base + 4)
    r = rows(state)
    assert len(r) == 1 and r[0]["msgs"] == 3 and r[0]["chatters"] == 2


def test_samples_capped_and_sanitized(cfg, state):
    c = mk(cfg, state)
    base = 1_700_000_000 // 5 * 5
    for i in range(10):
        c.add(base + 1, f"u{i}", f"@bob see https://x.io/{i} message number {i}")
    c.flush(now=base + 2)
    s = rows(state)[0]["samples"]
    assert len(s) == int(cfg.chat.sample_per_bucket)
    assert all("@" not in x and "http" not in x for x in s)


def test_no_identifiers_in_db_or_logs(cfg, state, caplog):
    caplog.set_level(logging.DEBUG)
    now = time.time()
    msgs = [(now, a, f"hello from me {i}") for i, a in enumerate(AUTHORS * 3)]
    def src(vid):
        yield from msgs
        time.sleep(5)

    c = mk(cfg, state, src, backoff=[0.05], max_seconds=0.6)
    assert c.run() == 0
    r = rows(state)
    assert sum(x["msgs"] for x in r) == 9 and sum(x["chatters"] for x in r) == 3
    dump = [tuple(x) for x in state._x("SELECT * FROM chat_buckets").fetchall()]
    dump += [tuple(x) for x in state._x("SELECT * FROM kv").fetchall()]
    text = repr(dump) + repr(r) + caplog.text
    for a in AUTHORS:
        assert a not in text
    log_dir = paths.DATA.logs
    for p in list(log_dir.glob("*")) + list(paths.DATA.heartbeat.glob("*")):
        if p.is_file():
            data = p.read_bytes().decode("utf-8", "ignore")
            assert not any(a in data for a in AUTHORS)
    assert "hello from me" not in caplog.text


def test_reconnects_with_backoff(cfg, state):
    calls = []

    def factory(vid):
        calls.append(time.monotonic())
        if len(calls) < 3:
            raise ConnectionError("boom")
        yield time.time(), "a", "omg"
        time.sleep(5)

    c = mk(cfg, state, factory, backoff=[0.2, 0.4], max_seconds=1.5)
    assert c.run() == 0
    assert len(calls) == 3
    assert calls[1] - calls[0] >= 0.18 and calls[2] - calls[1] >= 0.38
    assert sum(x["msgs"] for x in rows(state)) == 1


def test_exits_zero_on_recorder_done(cfg, state):
    state.kv_set("recorder.done", True)
    t0 = time.monotonic()
    c = mk(cfg, state, lambda vid: iter(lambda: time.sleep(10), None))
    assert c.run() == 0
    assert time.monotonic() - t0 < 3


def test_always_failing_source_warns_once_and_keeps_heartbeat(cfg, state, caplog):
    def factory(vid):
        raise RuntimeError("nope")
        yield  # pragma: no cover

    caplog.set_level(logging.INFO)
    c = mk(cfg, state, factory, backoff=[0.01], max_seconds=1.2)
    assert c.run() == 0
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    hb = heartbeat.read("chat")
    assert hb is not None and c.failures >= 5


def test_heartbeat_while_source_silent(cfg, state):
    c = mk(cfg, state, lambda vid: iter(lambda: time.sleep(10), None), max_seconds=0.5)
    c.run()
    assert heartbeat.read("chat") is not None


@pytest.mark.network
def test_real_stream_30s(state, request):
    """Reads ~30 s from a currently live stream via pytchat (set CHAT_TEST_VIDEO_ID, else the busiest GTA live found)."""
    if "network" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("network test: run with -m network")
    import os
    import subprocess
    from clipper.env import subprocess_env
    vid = os.environ.get("CHAT_TEST_VIDEO_ID")
    if not vid:
        out = subprocess.run([str(paths.ROOT / ".venv/Scripts/yt-dlp.exe"), "--cache-dir", str(paths.ROOT / ".cache/yt-dlp"),
                              "--flat-playlist", "--playlist-end", "20", "--print",
                              "%(id)s|%(live_status)s|%(concurrent_view_count)s",
                              "https://www.youtube.com/results?search_query=gta+live&sp=EgJAAQ%253D%253D"],
                             capture_output=True, text=True, env=subprocess_env(), timeout=120).stdout
        live = [l.split("|") for l in out.splitlines() if "|is_live|" in l]
        live = [(int(v) if v.isdigit() else 0, i) for i, _s, v in live]
        assert live, "no live stream found"
        vid = max(live)[1]
    cfg = load_config()
    c = Collector(cfg, state, vid, "s1", max_seconds=30)
    lags = []
    orig = c.add
    c.add = lambda ts, a, t: (lags.append(time.time() - ts), orig(ts, a, t))[1]
    c.run()
    n = sum(x["msgs"] for x in rows(state))
    print(f"\nCHAT_RESULT video={vid} msgs={n} per_min={n * 2} median_lag_s="
          f"{sorted(lags)[len(lags) // 2] if lags else None}")
    assert n > 0
