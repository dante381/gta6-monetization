"""publish_tick: called by the worker every loop. Plans and uploads what's due.

Order of checks each tick:
1. PAUSE kill switch, publish hold, quota block   -> return [] (reason logged once per change)
2. test_mode                                       -> no API at all; `would_upload` events
3. credentials (AuthRequired -> hold + attention)  -> plan -> upload one by one
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import config, log as elog
from ..models import QueueItem
from . import auth, scheduler, uploader
from .queue import load_sidecar

log = logging.getLogger("clipper.publish")

MAX_ATTEMPTS = 3
KV_REASON = "publish.last_reason"
KV_BLOCKED = "publish.blocked_until"
KV_REMINDER = "publish.private_reminder_day"


def _note(state: Any, reason: str | None, **fields: Any) -> None:
    """Log why publishing is idle, only when the reason changes."""
    prev = state.kv_get(KV_REASON)
    if prev == reason:
        return
    state.kv_set(KV_REASON, reason)
    if reason is None:
        log.info("publishing active (was: %s)", prev)
        elog.event("upload_skipped", reason="resumed", previous=prev)
    else:
        log.info("publishing idle: %s", reason)
        elog.event("upload_skipped", reason=reason, **fields)


def _blocked_until(state: Any) -> datetime | None:
    v = state.kv_get(KV_BLOCKED)
    if not v:
        return None
    try:
        return scheduler.parse_rfc3339(v)
    except Exception:
        return None


def _existing_slots(state: Any, now: datetime, test_mode: bool) -> list[QueueItem]:
    """Items that already hold a publish slot (count toward caps and the slot chain)."""
    out = [q for q in state.queue_items(status="scheduled") if q.publish_at]
    if test_mode:
        # simulated slots from earlier test-mode ticks, so the would-be schedule stays realistic;
        # only future ones, so switching test mode off doesn't inherit a fake backlog
        for q in state.queue_items(status="skipped", kind="publish"):
            if q.publish_at and scheduler.parse_rfc3339(q.publish_at) > now:
                out.append(q)
    return out


def _drop_missing(state: Any, items: list[QueueItem]) -> list[QueueItem]:
    ok = []
    for q in items:
        if not Path(q.clip_path).exists():
            state.queue_update(q.clip_id, status="failed", error="clip file missing")
            elog.event("upload_failed", clip_id=q.clip_id, item_kind=q.kind, error="clip file missing")
            continue
        ok.append(q)
    return ok


def _private_reminder(cfg: Any, state: Any, now: datetime) -> None:
    if cfg.publish.get("audit_passed", False):
        return
    day = scheduler.pacific_day(now)
    if state.kv_get(KV_REMINDER) == day:
        return
    state.kv_set(KV_REMINDER, day)
    log.warning("Reminder: until the YouTube API Services audit passes, YouTube keeps every "
                "API upload private (scheduled Shorts won't go public). Apply for the audit, then "
                "set publish.audit_passed: true to silence this.")


def _result(pu: scheduler.PlannedUpload, action: str, **extra: Any) -> dict:
    return {"clip_id": pu.item.clip_id, "kind": pu.kind, "action": action, "privacy": pu.privacy,
            "publish_at": pu.publish_at_str, "score": pu.item.final_score, "track": pu.item.track,
            **extra}


def publish_tick(cfg: Any, state: Any, now: datetime | None = None, youtube: Any = None) -> list[dict]:
    now = scheduler.ensure_aware(now) if now is not None else scheduler.utcnow()

    if config.paused(cfg):
        _note(state, "paused", file=str(config.kill_switch_path(cfg)))
        return []
    if state.publish_hold():
        _note(state, f"hold:{state.kv_get('hold.publish_reason') or ''}")
        return []
    blocked = _blocked_until(state)
    if blocked is not None and blocked > now:
        _note(state, "quota", until=scheduler.to_rfc3339(blocked))
        return []

    test_mode = bool(cfg.locked.get("test_mode", True))
    pending_pub = _drop_missing(state, state.queue_items(status="queued", kind="publish"))
    pending_bak = _drop_missing(state, state.queue_items(status="queued", kind="backup"))
    if not pending_pub and not (pending_bak and cfg.publish.get("upload_backups", True)):
        _note(state, None)
        return []

    tz = scheduler.local_tz(cfg)
    uploads_today = state.uploads_on(scheduler.pacific_day(now))
    plan = scheduler.plan_uploads(pending_pub, pending_bak, _existing_slots(state, now, test_mode),
                                  now, cfg, uploads_today, tz)
    if not plan:
        _note(state, "daily_upload_limit", uploads_today=uploads_today)
        return []
    _note(state, None)

    results: list[dict] = []
    if test_mode:
        for pu in plan:
            try:
                title = uploader.build_title(load_sidecar(pu.item.sidecar_path), cfg)
            except Exception:
                title = None
            state.queue_update(pu.item.clip_id, status="skipped", publish_at=pu.publish_at_str)
            elog.event("would_upload", clip_id=pu.item.clip_id, item_kind=pu.kind, privacy=pu.privacy,
                       publish_at=pu.publish_at_str, score=pu.item.final_score, track=pu.item.track,
                       title=title)
            results.append(_result(pu, "would_upload", title=title))
        return results

    if youtube is None:
        try:
            youtube = auth.build_youtube(auth.get_credentials())
        except auth.AuthRequired as e:
            state.set_publish_hold(True, "auth")
            elog.attention(str(e))
            _note(state, "hold:auth")
            return []
        except Exception as e:   # network trouble during refresh/build: just try next tick
            log.warning("YouTube client unavailable (%s: %s); will retry", type(e).__name__, e)
            return []

    _private_reminder(cfg, state, now)

    for pu in plan:
        if config.paused(cfg):
            _note(state, "paused", file=str(config.kill_switch_path(cfg)))
            break
        item = pu.item
        attempts = (item.attempts or 0) + 1
        try:
            sidecar = load_sidecar(item.sidecar_path)
            vid = uploader.upload(youtube, item, sidecar, cfg, publish_at=pu.publish_at,
                                  privacy=pu.privacy)
        except uploader.QuotaExceeded as e:
            until = scheduler.next_pacific_midnight(now)
            state.kv_set(KV_BLOCKED, scheduler.to_rfc3339(until))
            state.queue_update(item.clip_id, error=f"quota: {e}")
            log.warning("upload quota/rate limit hit (%s); uploads stop until %s", e,
                        scheduler.to_rfc3339(until))
            _note(state, "quota", until=scheduler.to_rfc3339(until))
            results.append(_result(pu, "blocked", error=str(e)))
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:500]
            status = "failed" if attempts >= MAX_ATTEMPTS else "queued"
            state.queue_update(item.clip_id, status=status, attempts=attempts, error=err,
                               publish_at=None)
            elog.event("upload_failed", clip_id=item.clip_id, item_kind=pu.kind, attempts=attempts,
                       gave_up=status == "failed", error=err)
            log.warning("upload %s failed (attempt %d/%d): %s", item.clip_id, attempts,
                        MAX_ATTEMPTS, err)
            results.append(_result(pu, "failed", error=err, attempts=attempts))
            continue

        status = "scheduled" if pu.publish_at else "uploaded"
        state.queue_update(item.clip_id, status=status, video_id=vid, publish_at=pu.publish_at_str,
                           attempts=attempts, error=None)
        state.record_upload(item.clip_id, scheduler.pacific_day(now), pu.kind, vid)
        elog.event("upload", clip_id=item.clip_id, item_kind=pu.kind, video_id=vid, privacy=pu.privacy,
                   publish_at=pu.publish_at_str, score=item.final_score, track=item.track,
                   title=uploader.build_title(sidecar, cfg))
        deleted = False
        if cfg.publish.get("delete_after_upload", False):
            try:
                Path(item.clip_path).unlink(missing_ok=True)
                deleted = True
            except OSError as e:
                log.warning("could not delete %s after upload: %s", item.clip_path, e)
        results.append(_result(pu, "uploaded", video_id=vid, deleted=deleted))
    return results
