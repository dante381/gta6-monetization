"""Slot planning, caps, rollover, Pacific-day helpers (pure functions)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clipper.config import load_config
from clipper.models import QueueItem
from clipper.publish import scheduler as sch

UTC = timezone.utc
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


def mk(clip_id, score, track="gameplay", kind="publish", publish_at=None, status="queued"):
    return QueueItem(clip_id=clip_id, clip_path=f"{clip_id}.mp4", sidecar_path=f"{clip_id}.json",
                     final_score=score, track=track, kind=kind, status=status,
                     publish_at=publish_at, created_at=1.0)


@pytest.fixture
def pcfg():
    return load_config(overrides={"publish": {"timezone": "UTC", "slot_hours": 2.4,
                                              "first_slot_delay_min": 30, "daily_upload_limit": 100,
                                              "upload_backups": True}})


def slots(pending, existing=(), slot_hours=2.4, delay=30, total=10, cut=3, now=NOW):
    return sch.assign_slots(pending, list(existing), now, slot_hours, delay, total, cut, UTC)


def test_slots_spacing_and_first_delay():
    out = slots([mk("a", 80), mk("b", 90), mk("c", 85)])
    assert [q.clip_id for q, _ in out] == ["b", "c", "a"]           # best first
    times = [t for _, t in out]
    assert times[0] == NOW + timedelta(minutes=30)
    assert times[1] - times[0] == timedelta(hours=2.4)
    assert times[2] - times[1] == timedelta(hours=2.4)


def test_slots_start_after_latest_scheduled():
    existing = [mk("x", 99, publish_at="2026-11-16T20:00:00Z", status="scheduled")]
    out = slots([mk("a", 80)], existing)
    assert out[0][1] == datetime(2026, 11, 16, 22, 24, tzinfo=UTC)


def test_past_scheduled_does_not_delay_first_slot():
    existing = [mk("x", 99, publish_at="2026-11-15T01:00:00Z", status="scheduled")]
    out = slots([mk("a", 80)], existing)
    assert out[0][1] == NOW + timedelta(minutes=30)


def test_daily_cap_10_and_rollover_next_day():
    items = [mk(f"c{i:02d}", 100 - i) for i in range(12)]
    out = slots(items, slot_hours=1.0)
    by_day: dict = {}
    for q, t in out:
        by_day.setdefault(t.date().isoformat(), []).append(q.clip_id)
    assert len(by_day["2026-11-16"]) == 10
    assert by_day["2026-11-17"] == ["c10", "c11"]                  # leftovers, best first
    assert out[10][1] == datetime(2026, 11, 17, 0, 0, tzinfo=UTC)


def test_rollover_counts_existing_by_publish_day():
    existing = [mk(f"e{i}", 90, publish_at=f"2026-11-16T{13 + i}:00:00Z", status="scheduled")
                for i in range(10)]
    # latest existing is 22:00; next slot 00:24 on the 17th
    out = slots([mk("lo", 76), mk("hi", 95)], existing)
    assert [q.clip_id for q, _ in out] == ["hi", "lo"]
    assert all(t.date().isoformat() == "2026-11-17" for _, t in out)


def test_cutscene_cap_3_per_day():
    items = [mk(f"cut{i}", 99 - i, track="cutscene") for i in range(5)]
    items += [mk(f"gp{i}", 80 - i) for i in range(3)]
    out = slots(items, slot_hours=1.0)
    day16 = [q for q, t in out if t.date().isoformat() == "2026-11-16"]
    day17 = [q for q, t in out if t.date().isoformat() == "2026-11-17"]
    assert sum(q.track == "cutscene" for q in day16) == 3
    assert {q.clip_id for q in day16 if q.track == "gameplay"} == {"gp0", "gp1", "gp2"}
    assert [q.clip_id for q in day17] == ["cut3", "cut4"]


def test_cutscene_cap_counts_existing():
    existing = [mk(f"e{i}", 90, track="cutscene", publish_at=f"2026-11-16T0{i}:00:00Z",
                   status="scheduled") for i in range(3)]
    out = slots([mk("cut", 99, track="cutscene"), mk("gp", 70)], existing)
    (first, t1), (second, t2) = out
    assert first.clip_id == "gp" and t1.date().isoformat() == "2026-11-16"
    assert second.clip_id == "cut" and t2.date().isoformat() == "2026-11-17"


def test_plan_orders_publish_then_backups(pcfg):
    pub = [mk("p1", 80), mk("p2", 90)]
    bak = [mk("b1", 55, kind="backup"), mk("b2", 70, kind="backup")]
    plan = sch.plan_uploads(pub, bak, [], NOW, pcfg, uploads_today=0, tz=UTC)
    assert [(p.item.clip_id, p.kind, p.privacy) for p in plan] == [
        ("p2", "publish", "private"), ("p1", "publish", "private"),
        ("b2", "backup", "unlisted"), ("b1", "backup", "unlisted")]
    assert plan[0].publish_at is not None and plan[2].publish_at is None


def test_plan_respects_upload_limit(pcfg):
    pub = [mk("p1", 80), mk("p2", 90)]
    bak = [mk("b1", 55, kind="backup")]
    plan = sch.plan_uploads(pub, bak, [], NOW, pcfg, uploads_today=98, tz=UTC)
    assert [p.item.clip_id for p in plan] == ["p2", "p1"]          # backups don't fit
    assert sch.plan_uploads(pub, bak, [], NOW, pcfg, uploads_today=100, tz=UTC) == []


def test_plan_upload_backups_false(pcfg):
    pcfg.publish.upload_backups = False
    plan = sch.plan_uploads([mk("p", 80)], [mk("b", 60, kind="backup")], [], NOW, pcfg, 0, UTC)
    assert [p.kind for p in plan] == ["publish"]


# ------------------------------------------------------------------ Pacific time

@pytest.mark.parametrize("use_fallback", [False, True])
def test_pacific_day(monkeypatch, use_fallback):
    if use_fallback:
        monkeypatch.setattr(sch, "_PACIFIC", None)
    # November (PST, UTC-8)
    assert sch.pacific_day(datetime(2026, 11, 17, 7, 59, tzinfo=UTC)) == "2026-11-16"
    assert sch.pacific_day(datetime(2026, 11, 17, 8, 0, tzinfo=UTC)) == "2026-11-17"
    # July (PDT, UTC-7)
    assert sch.pacific_day(datetime(2026, 7, 4, 6, 59, tzinfo=UTC)) == "2026-07-03"
    assert sch.pacific_day(datetime(2026, 7, 4, 7, 0, tzinfo=UTC)) == "2026-07-04"
    assert sch.next_pacific_midnight(datetime(2026, 11, 16, 12, 0, tzinfo=UTC)) == \
        datetime(2026, 11, 17, 8, 0, tzinfo=UTC)
    assert sch.next_pacific_midnight(datetime(2026, 7, 4, 12, 0, tzinfo=UTC)) == \
        datetime(2026, 7, 5, 7, 0, tzinfo=UTC)


def test_pacific_fallback_dst_boundaries():
    # 2026: DST starts Sun 8 March 10:00 UTC, ends Sun 1 November 09:00 UTC
    assert sch._pacific_offset(datetime(2026, 3, 8, 9, 59, tzinfo=UTC)) == timedelta(hours=-8)
    assert sch._pacific_offset(datetime(2026, 3, 8, 10, 0, tzinfo=UTC)) == timedelta(hours=-7)
    assert sch._pacific_offset(datetime(2026, 11, 1, 8, 59, tzinfo=UTC)) == timedelta(hours=-7)
    assert sch._pacific_offset(datetime(2026, 11, 1, 9, 0, tzinfo=UTC)) == timedelta(hours=-8)


def test_rfc3339_roundtrip():
    s = sch.to_rfc3339(datetime(2026, 11, 16, 12, 30, 15, 999, tzinfo=UTC))
    assert s == "2026-11-16T12:30:15Z"
    assert sch.parse_rfc3339(s) == datetime(2026, 11, 16, 12, 30, 15, tzinfo=UTC)
