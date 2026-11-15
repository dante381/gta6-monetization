"""publish_tick: called by the worker every loop. Plans and uploads what's due.

Order of checks each tick:
1. PAUSE kill switch, publish hold, quota block   -> return [] (reason logged once per change)
2. test_mode                                       -> no API at all; `would_upload` events
3. credentials (AuthRequired -> hold + attention)  -> plan -> upload one by one
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .. import config, log as elog, paths
from ..models import QueueItem
from . import auth, scheduler, uploader
from .queue import load_sidecar

log = logging.getLogger("clipper.publish")

MAX_ATTEMPTS = 3
KV_REASON = "publish.last_reason"
KV_BLOCKED = "publish.blocked_until"
KV_REMINDER = "publish.private_reminder_day"
KV_UPLOADING = "publish.uploading_since."      # + clip_id -> unix time the API call began
HOLDING = ("scheduled", "uploading", "needs_review")   # statuses that occupy a publish slot


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


def live_test_mode(cfg: Any) -> bool:
    """test_mode, re-read from config/locked.yaml on every call so editing the file stops a
    running worker. Safe direction only: True if the cfg says so OR the file says so OR the file is
    missing/unparseable/lacks the key. (cfg alone can still force test mode on, for tests.)"""
    cfg_val = bool(cfg.locked.get("test_mode", True))
    try:
        data = yaml.safe_load(Path(paths.LOCKED_YAML).read_text(encoding="utf-8"))
        file_val = bool(data["test_mode"])
    except Exception:
        file_val = True
    return cfg_val or file_val


def _recover_uploading(state: Any) -> None:
    """Rows left in 'uploading' mean the process died around the API call: the video may already
    exist on YouTube. Never re-upload automatically; park them for a human."""
    for q in state.queue_items(status="uploading"):
        since = state.kv_get(KV_UPLOADING + q.clip_id)
        state.queue_update(q.clip_id, status="needs_review",
                           error=f"upload interrupted (started {since}); check YouTube for a duplicate")
        elog.attention(f"clip {q.clip_id}: upload was interrupted and may already be on YouTube. "
                       f"Check the channel, then run resolve_needs_review(state, '{q.clip_id}', "
                       f"video_id=<id or None>). It will NOT be re-uploaded automatically.",
                       clip_id=q.clip_id)


def resolve_needs_review(state: Any, clip_id: str, video_id: str | None = None,
                         now: datetime | None = None) -> str:
    """Resolve a clip parked in 'needs_review' (interrupted upload).

    video_id given: the video exists on YouTube -> mark it uploaded (or scheduled when it has a
    publish slot) and record it in the upload counter. video_id None: it never reached YouTube ->
    put it back in the queue. Returns the new status.
    """
    q = state.queue_get(clip_id)
    if q is None:
        raise KeyError(f"no queue item {clip_id}")
    if q.status != "needs_review":
        raise ValueError(f"{clip_id} is '{q.status}', not needs_review")
    state.kv_set(KV_UPLOADING + clip_id, None)
    if video_id:
        status = "scheduled" if q.publish_at else "uploaded"
        state.queue_update(clip_id, status=status, video_id=video_id, error=None)
        state.record_upload(clip_id, scheduler.pacific_day(now or scheduler.utcnow()), q.kind, video_id)
        return status
    state.queue_update(clip_id, status="queued", publish_at=None, error=None)
    return "queued"


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
    out = [q for st in HOLDING for q in state.queue_items(status=st) if q.publish_at]
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

    _recover_uploading(state)
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

    test_mode = live_test_mode(cfg)
    pending_pub = _drop_missing(state, state.queue_items(status="queued", kind="publish"))
    pending_bak = _drop_missing(state, state.queue_items(status="queued", kind="backup"))
    if not pending_pub and not (pending_bak and cfg.publish.get("upload_backups", True)):
        _note(state, None)
        return []

    tz = scheduler.local_tz(cfg)
    # interrupted uploads may or may not have reached YouTube: count them as used
    uploads_today = (state.uploads_on(scheduler.pacific_day(now))
                     + len(state.queue_items(status="uploading"))
                     + len(state.queue_items(status="needs_review")))
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
        if live_test_mode(cfg):          # flipped on mid-run: stop before the next API call
            _note(state, "test_mode")
            break
        item = pu.item
        attempts = (item.attempts or 0) + 1
        try:
            sidecar = load_sidecar(item.sidecar_path)
            # journal BEFORE the API call so a crash can never lead to a silent duplicate
            state.kv_set(KV_UPLOADING + item.clip_id, time.time())
            state.queue_update(item.clip_id, status="uploading", publish_at=pu.publish_at_str)
            vid = uploader.upload(youtube, item, sidecar, cfg, publish_at=pu.publish_at,
                                  privacy=pu.privacy)
        except uploader.UploadPaused:
            state.queue_update(item.clip_id, status="queued", publish_at=None)
            state.kv_set(KV_UPLOADING + item.clip_id, None)
            _note(state, "paused", file=str(config.kill_switch_path(cfg)))
            break
        except uploader.QuotaExceeded as e:
            state.kv_set(KV_UPLOADING + item.clip_id, None)
            state.queue_update(item.clip_id, status="queued", publish_at=None)
            until = scheduler.next_pacific_midnight(now)
            state.kv_set(KV_BLOCKED, scheduler.to_rfc3339(until))
            state.queue_update(item.clip_id, error=f"quota: {e}")
            log.warning("upload quota/rate limit hit (%s); uploads stop until %s", e,
                        scheduler.to_rfc3339(until))
            _note(state, "quota", until=scheduler.to_rfc3339(until))
            results.append(_result(pu, "blocked", error=str(e)))
            break
        except Exception as e:
            state.kv_set(KV_UPLOADING + item.clip_id, None)
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
        state.kv_set(KV_UPLOADING + item.clip_id, None)
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


def drain(cfg: Any, state: Any, youtube: Any = None, max_ticks: int = 200,
          now_fn: Any = None) -> dict:
    """Run publish_tick until nothing is left to schedule/upload. Honours PAUSE, the publish
    hold, quota blocks and test_mode exactly like the worker (they make a tick return []).
    Returns {"results": [...], "ticks": n, "left": {status: count}, "idle_reason": str|None}."""
    results: list[dict] = []
    ticks = 0
    while ticks < max_ticks:
        ticks += 1
        batch = publish_tick(cfg, state, now=now_fn() if now_fn else None, youtube=youtube)
        results.extend(batch)
        if not batch or any(r["action"] == "blocked" for r in batch):
            break
    left: dict[str, int] = {}
    for st in ("queued", "uploading", "needs_review"):
        n = len(state.queue_items(status=st))
        if n:
            left[st] = n
    return {"results": results, "ticks": ticks, "left": left,
            "idle_reason": state.kv_get(KV_REASON)}


def main(argv: list[str] | None = None) -> int:
    """`python -m clipper publish`: publish whatever is queued, then exit (for clips left
    'queued' after the stream ended with PAUSE / a hold / quota in the way)."""
    import argparse
    from collections import Counter
    from ..state import StateDB

    ap = argparse.ArgumentParser(prog="clipper publish", description=main.__doc__)
    ap.add_argument("--max-ticks", type=int, default=200)
    args = ap.parse_args(argv)
    cfg = config.load_config()
    state = StateDB()
    try:
        out = drain(cfg, state, max_ticks=args.max_ticks)
    finally:
        state.close()
    actions = Counter(r["action"] for r in out["results"])
    print(f"publish: {out['ticks']} tick(s); " +
          (", ".join(f"{k}={v}" for k, v in sorted(actions.items())) or "nothing uploaded"))
    if out["left"]:
        print("still waiting: " + ", ".join(f"{k}={v}" for k, v in out["left"].items()))
    if out["idle_reason"]:
        print(f"idle reason: {out['idle_reason']}")
    if out["left"].get("needs_review"):
        print("needs_review clips: check YouTube, then resolve_needs_review(state, clip_id, video_id)")
    return 0
