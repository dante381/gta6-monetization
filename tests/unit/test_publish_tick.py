"""publish_tick end to end with a fake YouTube client. No real API calls, ever."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import httplib2
import pytest
from googleapiclient.errors import HttpError

from clipper import config, paths
from clipper.config import load_config
from clipper.models import ClipSidecar
from clipper.publish import auth, enqueue, publish_tick, uploader
from clipper.publish.scheduler import parse_rfc3339

UTC = timezone.utc
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)          # 04:00 Pacific, 16 Nov


# ------------------------------------------------------------------ fakes / helpers

def http_error(status: int, reason: str | None = None) -> HttpError:
    resp = httplib2.Response({"status": status})
    err = {"code": status, "message": reason or "error"}
    if reason:
        err["errors"] = [{"reason": reason, "message": reason}]
    return HttpError(resp, json.dumps({"error": err}).encode("utf-8"))


class FakeRequest:
    def __init__(self, yt: "FakeYouTube", call: dict):
        self.yt, self.call, self.chunks = yt, call, 0

    def next_chunk(self):
        self.yt.next_chunk_calls += 1
        if self.yt.fail_always is not None:
            raise self.yt.fail_always
        if self.yt.errors:
            raise self.yt.errors.pop(0)
        self.chunks += 1
        if self.chunks < self.yt.chunks_per_upload:
            return object(), None
        self.yt.counter += 1
        vid = f"vid{self.yt.counter}"
        self.call["video_id"] = vid
        return None, {"id": vid}


class FakeVideos:
    def __init__(self, yt):
        self.yt = yt

    def insert(self, part, body, media_body):
        call = {"part": part, "body": body, "media": media_body}
        self.yt.inserts.append(call)
        return FakeRequest(self.yt, call)


class FakeYouTube:
    def __init__(self, chunks_per_upload: int = 2):
        self.inserts: list[dict] = []
        self.errors: list[Exception] = []
        self.fail_always: Exception | None = None
        self.chunks_per_upload = chunks_per_upload
        self.counter = 0
        self.next_chunk_calls = 0

    def videos(self):
        return FakeVideos(self)


def make_clip(tmp_path, clip_id: str, score: float, route: str = "publish",
              track: str = "gameplay", **kw) -> ClipSidecar:
    d = tmp_path / "clips"
    d.mkdir(exist_ok=True)
    mp4 = d / f"{clip_id}.mp4"
    mp4.write_bytes(b"\x00" * 2048)
    sc = ClipSidecar(clip_id=clip_id, clip_path=str(mp4), track=track, route=route,
                     final_score=score, signal_score=score,
                     title=kw.get("title", f"Insane chase {clip_id}"),
                     description=kw.get("description", "Wild police chase in Vice City."),
                     hashtags=kw.get("hashtags", ["#GTA6", "#ViceCity"]), hook="no way",
                     spoiler=kw.get("spoiler", False), duration_s=30.0, start_s=0.0, end_s=30.0,
                     created_at=time.time())
    (d / f"{clip_id}.json").write_text(json.dumps(sc.to_dict()), encoding="utf-8")
    return sc


def events(kind: str | None = None) -> list[dict]:
    p = paths.DATA.logs / "clips.jsonl"
    if not p.exists():
        return []
    out = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [e for e in out if kind is None or e["kind"] == kind]


@pytest.fixture
def pcfg(tmp_path, monkeypatch):
    """Real config, test_mode off (cfg AND a temp locked.yaml, which publish_tick re-reads live),
    UTC days, and a private kill-switch file per test."""
    c = load_config(overrides={"publish": {"timezone": "UTC", "slot_hours": 2.4,
                                           "first_slot_delay_min": 30, "upload_backups": True,
                                           "daily_upload_limit": 100,
                                           "delete_after_upload": False}})
    live = tmp_path / "locked_live.yaml"          # loaded config above used the real file
    live.write_text("test_mode: false\n", encoding="utf-8")
    monkeypatch.setattr(paths, "LOCKED_YAML", live)
    c.locked.test_mode = False
    c.locked.kill_switch_file = f".tmp/PAUSE_pytest_{uuid.uuid4().hex}"
    yield c
    config.kill_switch_path(c).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def no_real_api(monkeypatch):
    """Safety net: building a real client or sleeping in backoff is never allowed here."""
    def boom(*a, **k):
        raise AssertionError("real YouTube client must never be built in tests")
    monkeypatch.setattr(auth, "build_youtube", boom)
    monkeypatch.setattr(uploader, "_sleep", lambda s: None)


@pytest.fixture
def yt():
    return FakeYouTube()


# ------------------------------------------------------------------ tests

def test_enqueue_kinds(tmp_path, state, pcfg):
    a = enqueue(make_clip(tmp_path, "a", 80), state, pcfg)
    b = enqueue(make_clip(tmp_path, "b", 60, route="backup", track="cutscene"), state, pcfg)
    assert enqueue(make_clip(tmp_path, "c", 30, route="drop"), state, pcfg) is None
    assert (a.kind, a.status, a.track, a.final_score) == ("publish", "queued", "gameplay", 80)
    assert (b.kind, b.track) == ("backup", "cutscene")
    assert b.sidecar_path.endswith("b.json")
    assert {q.clip_id for q in state.queue_items()} == {"a", "b"}


def test_uploads_scheduled_then_backups(tmp_path, state, pcfg, yt):
    enqueue(make_clip(tmp_path, "p1", 80), state, pcfg)
    enqueue(make_clip(tmp_path, "p2", 92), state, pcfg)
    enqueue(make_clip(tmp_path, "b1", 60, route="backup"), state, pcfg)
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert [(r["clip_id"], r["action"]) for r in res] == [
        ("p2", "uploaded"), ("p1", "uploaded"), ("b1", "uploaded")]
    statuses = [c["body"]["status"] for c in yt.inserts]
    assert [s["privacyStatus"] for s in statuses] == ["private", "private", "unlisted"]
    assert "publishAt" in statuses[0] and "publishAt" not in statuses[2]
    q = state.queue_get("p2")
    assert (q.status, q.video_id, q.attempts) == ("scheduled", "vid1", 1)
    assert parse_rfc3339(q.publish_at) == datetime(2026, 11, 16, 12, 30, tzinfo=UTC)
    assert state.queue_get("b1").status == "uploaded"
    assert state.uploads_on("2026-11-16") == 3
    assert len(events("upload")) == 3
    # files stay (delete_after_upload false); a second tick has nothing to do
    assert (tmp_path / "clips" / "p2.mp4").exists()
    assert publish_tick(pcfg, state, now=NOW, youtube=yt) == []
    assert len(yt.inserts) == 3


def test_second_tick_continues_slot_chain(tmp_path, state, pcfg, yt):
    enqueue(make_clip(tmp_path, "p1", 80), state, pcfg)
    publish_tick(pcfg, state, now=NOW, youtube=yt)
    enqueue(make_clip(tmp_path, "p2", 99), state, pcfg)
    publish_tick(pcfg, state, now=NOW, youtube=yt)
    t1 = parse_rfc3339(state.queue_get("p1").publish_at)
    t2 = parse_rfc3339(state.queue_get("p2").publish_at)
    assert (t2 - t1).total_seconds() == 2.4 * 3600


def test_upload_backups_false(tmp_path, state, pcfg, yt):
    pcfg.publish.upload_backups = False
    enqueue(make_clip(tmp_path, "b1", 60, route="backup"), state, pcfg)
    assert publish_tick(pcfg, state, now=NOW, youtube=yt) == []
    assert yt.inserts == [] and state.queue_get("b1").status == "queued"


def test_daily_cap_through_tick(tmp_path, state, pcfg, yt):
    pcfg.publish.slot_hours = 1.0
    for i in range(12):
        enqueue(make_clip(tmp_path, f"c{i:02d}", 99 - i), state, pcfg)
    publish_tick(pcfg, state, now=NOW, youtube=yt)
    days = [parse_rfc3339(q.publish_at).date().isoformat()
            for q in state.queue_items(status="scheduled")]
    assert days.count("2026-11-16") == 10 and days.count("2026-11-17") == 2


def test_pacific_day_upload_limit(tmp_path, state, pcfg, yt):
    for i in range(100):
        state.record_upload(f"old{i}", "2026-11-16", "publish", f"x{i}")
    enqueue(make_clip(tmp_path, "p1", 80), state, pcfg)
    late_pacific = datetime(2026, 11, 17, 7, 30, tzinfo=UTC)   # 23:30 PT on the 16th
    assert publish_tick(pcfg, state, now=late_pacific, youtube=yt) == []
    assert yt.inserts == []
    next_day = datetime(2026, 11, 17, 8, 30, tzinfo=UTC)       # 00:30 PT on the 17th
    res = publish_tick(pcfg, state, now=next_day, youtube=yt)
    assert [r["action"] for r in res] == ["uploaded"]
    assert state.uploads_on("2026-11-17") == 1


def test_pause_file_blocks_everything(tmp_path, state, pcfg, yt):
    enqueue(make_clip(tmp_path, "p1", 80), state, pcfg)
    ks = config.kill_switch_path(pcfg)
    ks.parent.mkdir(parents=True, exist_ok=True)
    ks.write_text("", encoding="utf-8")
    try:
        assert config.paused(pcfg)
        assert publish_tick(pcfg, state, now=NOW, youtube=yt) == []
        assert publish_tick(pcfg, state, now=NOW, youtube=yt) == []
        assert yt.inserts == []
        assert [e["reason"] for e in events("upload_skipped")] == ["paused"]   # logged once
    finally:
        ks.unlink(missing_ok=True)
    assert [r["action"] for r in publish_tick(pcfg, state, now=NOW, youtube=yt)] == ["uploaded"]


def test_pause_rechecked_before_each_upload(tmp_path, state, pcfg, yt, monkeypatch):
    enqueue(make_clip(tmp_path, "p1", 90), state, pcfg)
    enqueue(make_clip(tmp_path, "p2", 80), state, pcfg)
    real_upload = uploader.upload

    def upload_then_pause(*a, **k):
        vid = real_upload(*a, **k)
        config.kill_switch_path(pcfg).parent.mkdir(parents=True, exist_ok=True)
        config.kill_switch_path(pcfg).write_text("", encoding="utf-8")
        return vid
    monkeypatch.setattr(uploader, "upload", upload_then_pause)
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert [r["clip_id"] for r in res] == ["p1"]
    assert state.queue_get("p2").status == "queued"


def test_default_kill_switch_is_project_root(cfg):
    assert config.kill_switch_path(cfg) == paths.ROOT / "PAUSE"


def test_publish_hold_blocks(tmp_path, state, pcfg, yt):
    enqueue(make_clip(tmp_path, "p1", 80), state, pcfg)
    state.set_publish_hold(True, "repair")
    assert publish_tick(pcfg, state, now=NOW, youtube=yt) == []
    assert yt.inserts == []
    state.set_publish_hold(False)
    assert len(publish_tick(pcfg, state, now=NOW, youtube=yt)) == 1


def test_test_mode_would_upload_never_calls_api(tmp_path, state, pcfg, monkeypatch):
    pcfg.locked.test_mode = True

    def no_creds(*a, **k):
        raise AssertionError("test mode must not touch credentials")
    monkeypatch.setattr(auth, "get_credentials", no_creds)
    yt = FakeYouTube()
    enqueue(make_clip(tmp_path, "p1", 80), state, pcfg)
    enqueue(make_clip(tmp_path, "b1", 60, route="backup"), state, pcfg)
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert yt.inserts == [] and yt.next_chunk_calls == 0
    assert [r["action"] for r in res] == ["would_upload", "would_upload"]
    ev = events("would_upload")
    assert [(e["clip_id"], e["privacy"]) for e in ev] == [("p1", "private"), ("b1", "unlisted")]
    assert ev[0]["publish_at"] == "2026-11-16T12:30:00Z" and ev[1]["publish_at"] is None
    assert "GTA 6" in ev[0]["title"]
    assert state.queue_get("p1").status == "skipped"
    assert state.uploads_on("2026-11-16") == 0
    # no API, no default client either
    assert publish_tick(pcfg, state, now=NOW, youtube=None) == []


def test_auth_required_sets_hold_and_attention(tmp_path, state, pcfg, monkeypatch):
    monkeypatch.setattr(paths, "TOKEN", tmp_path / "secrets" / "token.json")   # missing
    enqueue(make_clip(tmp_path, "p1", 80), state, pcfg)
    assert publish_tick(pcfg, state, now=NOW, youtube=None) == []
    assert state.publish_hold() is True
    assert state.kv_get("hold.publish_reason") == "auth"
    att = events("attention")
    assert att and "python -m clipper auth" in att[0]["message"]
    assert "Publish app" in att[0]["message"]
    assert (paths.DATA.logs / "ATTENTION.txt").exists()
    assert state.queue_get("p1").status == "queued"


def test_retry_on_500(tmp_path, state, pcfg, yt):
    yt.errors = [http_error(500), http_error(503), ConnectionError("reset")]
    enqueue(make_clip(tmp_path, "p1", 80), state, pcfg)
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert [r["action"] for r in res] == ["uploaded"]
    assert len(yt.inserts) == 1                     # resumed, not restarted
    assert state.queue_get("p1").attempts == 1


def test_quota_error_blocks_for_the_day(tmp_path, state, pcfg, yt):
    yt.errors = [http_error(403, "uploadLimitExceeded")]
    enqueue(make_clip(tmp_path, "p1", 90), state, pcfg)
    enqueue(make_clip(tmp_path, "p2", 80), state, pcfg)
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert [r["action"] for r in res] == ["blocked"]
    assert len(yt.inserts) == 1                     # p2 never tried
    assert state.kv_get("publish.blocked_until") == "2026-11-17T08:00:00Z"   # next Pacific midnight
    q = state.queue_get("p1")
    assert q.status == "queued" and q.attempts == 0
    later = datetime(2026, 11, 17, 7, 0, tzinfo=UTC)
    assert publish_tick(pcfg, state, now=later, youtube=yt) == []
    assert len(yt.inserts) == 1
    after = datetime(2026, 11, 17, 8, 5, tzinfo=UTC)
    assert len(publish_tick(pcfg, state, now=after, youtube=yt)) == 2


def test_failure_on_one_item_does_not_stop_tick(tmp_path, state, pcfg, yt):
    yt.errors = [http_error(400, "invalidTitle")]
    enqueue(make_clip(tmp_path, "p1", 90), state, pcfg)
    enqueue(make_clip(tmp_path, "p2", 80), state, pcfg)
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert [(r["clip_id"], r["action"]) for r in res] == [("p1", "failed"), ("p2", "uploaded")]
    q = state.queue_get("p1")
    assert (q.status, q.attempts) == ("queued", 1) and "400" in q.error


def test_give_up_after_3_attempts(tmp_path, state, pcfg, yt):
    yt.fail_always = http_error(400, "invalidMetadata")
    enqueue(make_clip(tmp_path, "p1", 80), state, pcfg)
    for _ in range(3):
        publish_tick(pcfg, state, now=NOW, youtube=yt)
    q = state.queue_get("p1")
    assert (q.status, q.attempts) == ("failed", 3)
    assert publish_tick(pcfg, state, now=NOW, youtube=yt) == []
    assert len(yt.inserts) == 3
    assert events("upload_failed")[-1]["gave_up"] is True


def test_delete_after_upload(tmp_path, state, pcfg, yt):
    pcfg.publish.delete_after_upload = True
    sc = make_clip(tmp_path, "p1", 80)
    enqueue(sc, state, pcfg)
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert res[0]["deleted"] is True
    assert not (tmp_path / "clips" / "p1.mp4").exists()
    assert (tmp_path / "clips" / "p1.json").exists()


def test_missing_clip_marked_failed(tmp_path, state, pcfg, yt):
    sc = make_clip(tmp_path, "p1", 80)
    enqueue(sc, state, pcfg)
    (tmp_path / "clips" / "p1.mp4").unlink()
    assert publish_tick(pcfg, state, now=NOW, youtube=yt) == []
    assert state.queue_get("p1").status == "failed" and yt.inserts == []


def test_private_reminder_once_per_day(tmp_path, state, pcfg, yt, caplog):
    enqueue(make_clip(tmp_path, "p1", 80), state, pcfg)
    enqueue(make_clip(tmp_path, "p2", 70), state, pcfg)
    with caplog.at_level("WARNING", logger="clipper.publish"):
        publish_tick(pcfg, state, now=NOW, youtube=yt)
        enqueue(make_clip(tmp_path, "p3", 85), state, pcfg)
        publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert sum("audit" in r.getMessage() for r in caplog.records) == 1
