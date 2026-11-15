"""Publish robustness: live test_mode, crash journal / needs_review, mid-upload PAUSE, drain CLI."""
from __future__ import annotations

import pytest

from clipper import config, paths
from clipper.publish import enqueue, publish_tick, publisher, uploader
from clipper.publish.queue import load_sidecar
from tests.unit.test_publish_tick import (FakeYouTube, NOW, events, make_clip, no_real_api,  # noqa: F401
                                          pcfg, yt)


def _q(tmp_path, state, pcfg, cid, score=80, **kw):
    return enqueue(make_clip(tmp_path, cid, score, **kw), state, pcfg)


def _wrap_insert(yt, hook):
    """Call hook(kwargs, request) -> optional replacement request, on every videos().insert."""
    orig = yt.videos

    def videos():
        v = orig()
        ins = v.insert

        def insert(**kw):
            hook(kw)
            return ins(**kw)
        v.insert = insert
        return v
    yt.videos = videos


# ---- M3: live test_mode
def test_locked_yaml_flip_stops_uploads(tmp_path, state, pcfg, yt):
    for i in "abc":
        _q(tmp_path, state, pcfg, i)
    _wrap_insert(yt, lambda kw: paths.LOCKED_YAML.write_text("test_mode: true\n", encoding="utf-8"))
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert [r["action"] for r in res] == ["uploaded"]          # 2nd/3rd never reach the API
    assert len(yt.inserts) == 1
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert {r["action"] for r in res} == {"would_upload"} and len(yt.inserts) == 1


@pytest.mark.parametrize("content", ["", "{{{ not yaml", "other: 1\n"])
def test_missing_or_bad_locked_file_means_test_mode(tmp_path, state, pcfg, yt, content):
    paths.LOCKED_YAML.write_text(content, encoding="utf-8")
    _q(tmp_path, state, pcfg, "a")
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert [r["action"] for r in res] == ["would_upload"] and not yt.inserts
    paths.LOCKED_YAML.unlink()
    assert publisher.live_test_mode(pcfg) is True


def test_cfg_override_forces_test_mode(tmp_path, state, pcfg, yt):
    pcfg.locked.test_mode = True        # file says false; either True wins
    _q(tmp_path, state, pcfg, "a")
    assert [r["action"] for r in publish_tick(pcfg, state, now=NOW, youtube=yt)] == ["would_upload"]


# ---- M6: journal, crash, needs_review, caps
def test_status_uploading_during_api_call(tmp_path, state, pcfg, yt):
    _q(tmp_path, state, pcfg, "a")
    seen = {}
    _wrap_insert(yt, lambda kw: seen.update(status=state.queue_get("a").status))
    publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert seen["status"] == "uploading"
    assert state.queue_get("a").status == "scheduled"


def test_crash_after_upload_never_reuploads(tmp_path, state, pcfg, yt):
    _q(tmp_path, state, pcfg, "a")
    _q(tmp_path, state, pcfg, "b", 70)

    def die(kw):
        raise KeyboardInterrupt("simulated process death after the API call")
    # a crash between upload() returning and the queue update leaves the row in 'uploading'
    orig_update = state.queue_update

    def update(clip_id, **f):
        if f.get("status") == "scheduled":
            raise KeyboardInterrupt("simulated process death")
        return orig_update(clip_id, **f)
    state.queue_update = update
    with pytest.raises(KeyboardInterrupt):
        publish_tick(pcfg, state, now=NOW, youtube=yt)
    state.queue_update = orig_update
    assert state.queue_get("a").status == "uploading" and len(yt.inserts) == 1

    res = publish_tick(pcfg, state, now=NOW, youtube=yt)       # "restart"
    assert state.queue_get("a").status == "needs_review"
    assert all(r["clip_id"] != "a" for r in res)
    assert [c["video_id"] for c in yt.inserts[1:]] == ["vid2"]   # only b went up, a never again
    assert any(e["kind"] == "attention" and "a" in e["message"] for e in events())


def test_needs_review_counts_against_caps(tmp_path, state, pcfg, yt):
    pcfg.locked.caps.daily_total = 2
    _q(tmp_path, state, pcfg, "stuck", 95)
    state.queue_update("stuck", status="needs_review", publish_at="2026-11-16T12:30:00Z")
    for i in "xyz":
        _q(tmp_path, state, pcfg, i, 80)
    publish_tick(pcfg, state, now=NOW, youtube=yt)
    day16 = [q for q in state.queue_items(status="scheduled") if q.publish_at.startswith("2026-11-16")]
    assert len(day16) == 1                                  # 2 slots - 1 held by needs_review


def test_needs_review_reduces_upload_budget(tmp_path, state, pcfg, yt):
    _q(tmp_path, state, pcfg, "stuck", 95)
    state.queue_update("stuck", status="needs_review")
    for i in "xy":
        _q(tmp_path, state, pcfg, i, 80)
    pcfg.publish.daily_upload_limit = 2
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert len(res) == 1


def test_resolve_needs_review(tmp_path, state, pcfg, yt):
    _q(tmp_path, state, pcfg, "a")
    _q(tmp_path, state, pcfg, "b")
    state.queue_update("a", status="needs_review", publish_at="2026-11-17T10:00:00Z")
    state.queue_update("b", status="needs_review")
    assert publisher.resolve_needs_review(state, "a", video_id="VID9", now=NOW) == "scheduled"
    assert state.queue_get("a").video_id == "VID9"
    assert state.uploads_on("2026-11-16") == 1
    assert publisher.resolve_needs_review(state, "b") == "queued"
    assert state.queue_get("b").publish_at is None
    with pytest.raises(ValueError):
        publisher.resolve_needs_review(state, "b")


# ---- PAUSE mid-upload
def test_pause_mid_upload_aborts_cleanly(tmp_path, state, pcfg):
    _q(tmp_path, state, pcfg, "a")
    yt = FakeYouTube(chunks_per_upload=3)
    orig = yt.videos

    def videos():
        v = orig()
        ins = v.insert

        def insert(**kw):
            req = ins(**kw)
            real = req.next_chunk

            def nxt():
                r = real()
                config.kill_switch_path(pcfg).write_text("x")      # PAUSE appears after chunk 1
                return r
            req.next_chunk = nxt
            return req
        v.insert = insert
        return v
    yt.videos = videos
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    q = state.queue_get("a")
    assert res == [] and q.status == "queued" and q.attempts == 0 and q.publish_at is None
    assert yt.next_chunk_calls == 1 and state.uploads_on("2026-11-16") == 0


def test_uploader_raises_uploadpaused(tmp_path, state, pcfg, yt):
    item = _q(tmp_path, state, pcfg, "a")
    config.kill_switch_path(pcfg).write_text("x")
    with pytest.raises(uploader.UploadPaused):
        uploader.upload(yt, item, load_sidecar(item.sidecar_path), pcfg)


# ---- M4: drain / CLI
def test_drain_publishes_everything_queued(tmp_path, state, pcfg, yt):
    for i in "abcd":
        _q(tmp_path, state, pcfg, i)
    out = publisher.drain(pcfg, state, youtube=yt, now_fn=lambda: NOW)
    assert len(out["results"]) == 4 and out["left"] == {}


def test_drain_respects_pause_and_hold(tmp_path, state, pcfg, yt):
    _q(tmp_path, state, pcfg, "a")
    config.kill_switch_path(pcfg).write_text("x")
    out = publisher.drain(pcfg, state, youtube=yt, now_fn=lambda: NOW)
    assert out["results"] == [] and out["left"] == {"queued": 1} and out["idle_reason"] == "paused"
    config.kill_switch_path(pcfg).unlink()
    state.set_publish_hold(True, "repair")
    out = publisher.drain(pcfg, state, youtube=yt, now_fn=lambda: NOW)
    assert out["left"] == {"queued": 1} and not yt.inserts


def test_drain_test_mode_terminates(tmp_path, state, pcfg, yt):
    pcfg.locked.test_mode = True
    for i in "ab":
        _q(tmp_path, state, pcfg, i)
    out = publisher.drain(pcfg, state, youtube=yt, now_fn=lambda: NOW)
    assert {r["action"] for r in out["results"]} == {"would_upload"} and not yt.inserts


def test_main_prints_summary(tmp_path, state, pcfg, monkeypatch, capsys):
    pcfg.locked.test_mode = True
    _q(tmp_path, state, pcfg, "a")
    monkeypatch.setattr(config, "load_config", lambda *a, **k: pcfg)
    monkeypatch.setattr("clipper.state.StateDB", lambda: state)
    monkeypatch.setattr(state, "close", lambda: None)
    assert publisher.main([]) == 0
    assert "would_upload=1" in capsys.readouterr().out
