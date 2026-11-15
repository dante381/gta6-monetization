"""Uploader: metadata body, resumable media, retries and quota errors (fake client)."""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from clipper.config import load_config
from clipper.publish import uploader
from clipper.publish.queue import enqueue
from tests.unit.test_publish_tick import FakeYouTube, http_error, make_clip

UTC = timezone.utc
SLOT = datetime(2026, 11, 16, 14, 54, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(uploader, "_sleep", slept.append)
    return slept


@pytest.fixture
def ucfg():
    return load_config()


def _item(tmp_path, state, ucfg, **kw):
    sc = make_clip(tmp_path, kw.pop("clip_id", "c1"), kw.pop("score", 80), **kw)
    return enqueue(sc, state, ucfg), sc


def test_scheduled_body(tmp_path, state, ucfg):
    item, sc = _item(tmp_path, state, ucfg, title="Cops lose me in the swamp")
    body = uploader.build_body(item, sc, ucfg, publish_at=SLOT)
    sn, st = body["snippet"], body["status"]
    assert "GTA 6" in sn["title"] and len(sn["title"]) <= 100
    assert sn["description"].rstrip().endswith(" ".join(uploader.build_hashtags(sc, ucfg)))
    assert "#Shorts" in sn["description"].split("\n")[-1]
    assert all(not t.startswith("#") for t in sn["tags"]) and "Shorts" in sn["tags"]
    assert sn["categoryId"] == "20" and sn["defaultLanguage"] == "en"
    assert st == {"selfDeclaredMadeForKids": False, "privacyStatus": "private",
                  "publishAt": "2026-11-16T14:54:00Z"}
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", st["publishAt"])


def test_backup_body_unlisted(tmp_path, state, ucfg):
    item, sc = _item(tmp_path, state, ucfg, route="backup")
    st = uploader.build_body(item, sc, ucfg, privacy="unlisted")["status"]
    assert st["privacyStatus"] == "unlisted" and "publishAt" not in st
    assert st["selfDeclaredMadeForKids"] is False


def test_title_rules(tmp_path, state, ucfg):
    item, sc = _item(tmp_path, state, ucfg, title="x" * 150 + " <wow>", spoiler=True,
                     track="cutscene", hashtags=["Shorts", "#GTA 6 ", "#gta6"])
    title = uploader.build_title(sc, ucfg)
    assert len(title) <= 100 and title.endswith("| GTA 6") and "<" not in title
    sc.title = ""
    assert "GTA 6" in uploader.build_title(sc, ucfg)
    tags = uploader.build_hashtags(sc, ucfg)
    assert [t.lower() for t in tags].count("#shorts") == 1
    assert [t.lower() for t in tags].count("#gta6") == 1


def test_media_upload_is_resumable_8mb(tmp_path, state, ucfg):
    item, sc = _item(tmp_path, state, ucfg)
    yt = FakeYouTube(chunks_per_upload=3)
    vid = uploader.upload(yt, item, sc, ucfg, publish_at=SLOT)
    assert vid == "vid1" and yt.next_chunk_calls == 3
    call = yt.inserts[0]
    assert call["part"] == "snippet,status"
    m = call["media"]
    assert m.resumable() is True and m.chunksize() == 8 * 1024 * 1024
    assert m.mimetype() == "video/mp4"


def test_retry_then_give_up_after_5(tmp_path, state, ucfg, fast_sleep):
    item, sc = _item(tmp_path, state, ucfg)
    yt = FakeYouTube()
    yt.fail_always = http_error(502)
    with pytest.raises(uploader.UploadError) as ei:
        uploader.upload(yt, item, sc, ucfg)
    assert not isinstance(ei.value, uploader.QuotaExceeded)
    assert yt.next_chunk_calls == 6                     # 1 try + 5 retries
    assert sum(fast_sleep) > 0 and max(fast_sleep) <= 5.0   # sleeps in <=5 s steps


@pytest.mark.parametrize("status,reason", [(403, "quotaExceeded"), (403, "uploadLimitExceeded"),
                                           (403, "rateLimitExceeded"), (429, None)])
def test_quota_errors(tmp_path, state, ucfg, status, reason):
    item, sc = _item(tmp_path, state, ucfg)
    yt = FakeYouTube()
    yt.errors = [http_error(status, reason)]
    with pytest.raises(uploader.QuotaExceeded):
        uploader.upload(yt, item, sc, ucfg)
    assert yt.next_chunk_calls == 1


def test_other_403_is_plain_failure(tmp_path, state, ucfg):
    item, sc = _item(tmp_path, state, ucfg)
    yt = FakeYouTube()
    yt.errors = [http_error(403, "forbidden")]
    with pytest.raises(uploader.UploadError) as ei:
        uploader.upload(yt, item, sc, ucfg)
    assert not isinstance(ei.value, uploader.QuotaExceeded)
