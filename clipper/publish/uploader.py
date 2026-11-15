"""One resumable YouTube upload (videos.insert) with retries.

Scheduled Shorts go up as private with status.publishAt (YouTube flips them public at that
time). Backups go up as unlisted. Until the YouTube API audit passes, YouTube forces every
API upload to private no matter what we ask for.
"""
from __future__ import annotations

import json
import logging
import random
import socket
import time
from datetime import datetime
from typing import Any, Callable

from ..models import ClipSidecar, QueueItem
from .scheduler import to_rfc3339

log = logging.getLogger("clipper.publish.uploader")

CHUNK_SIZE = 8 * 1024 * 1024
MAX_RETRIES = 5
RETRY_STATUS = {500, 502, 503, 504}
QUOTA_REASONS = {"quotaExceeded", "uploadLimitExceeded", "dailyLimitExceeded",
                 "rateLimitExceeded", "userRateLimitExceeded"}
TITLE_MAX = 100
DESCRIPTION_MAX = 4900          # API limit is 5000 bytes; keep a margin
TAGS_MAX_CHARS = 450            # API limit is 500 characters in total
_sleep = time.sleep             # tests replace this


class UploadError(RuntimeError):
    """The upload failed for this item (counts as one attempt)."""


class UploadPaused(Exception):
    """The PAUSE kill switch appeared mid-upload: the upload was abandoned cleanly. It is not a
    failure (no attempt is counted) and the item stays queued."""


class QuotaExceeded(UploadError):
    """Daily quota / upload limit / rate limit hit: stop uploading until the next Pacific day."""


# --------------------------------------------------------------------- metadata

def _clean(s: str) -> str:
    # YouTube rejects '<' and '>' in titles and descriptions
    return (s or "").replace("<", "").replace(">", "").strip()


def build_title(sidecar: ClipSidecar, cfg: Any) -> str:
    game = cfg.publish.game_name
    title = " ".join(_clean(sidecar.title).split())
    if not title:
        title = f"{game} moment"
    if sidecar.spoiler and "spoiler" not in title.lower():
        title = f"{title} (spoiler)"
    if game.lower() not in title.lower():
        suffix = f" | {game}"
        title = title[: TITLE_MAX - len(suffix)].rstrip() + suffix
    if len(title) > TITLE_MAX:
        title = title[:TITLE_MAX].rstrip()
        if game.lower() not in title.lower():   # truncation cut the game name off
            suffix = f" | {game}"
            title = title[: TITLE_MAX - len(suffix)].rstrip() + suffix
    return title


def build_hashtags(sidecar: ClipSidecar, cfg: Any) -> list[str]:
    raw = list(sidecar.hashtags or []) + list(cfg.publish.get("base_hashtags") or [])
    out: list[str] = []
    seen: set[str] = set()
    for h in raw:
        h = "".join(str(h).split()).lstrip("#")
        h = "".join(ch for ch in h if ch.isalnum() or ch == "_")
        if not h or h.lower() in seen:
            continue
        seen.add(h.lower())
        out.append("#" + h)
    if "shorts" not in seen:
        out.insert(0, "#Shorts")
    return out


def build_body(item: QueueItem, sidecar: ClipSidecar, cfg: Any,
               publish_at: datetime | str | None = None, privacy: str = "private") -> dict:
    p = cfg.publish
    hashtags = build_hashtags(sidecar, cfg)
    tag_line = " ".join(hashtags)
    text = _clean(sidecar.description)
    room = DESCRIPTION_MAX - len(tag_line.encode("utf-8")) - 2
    if len(text.encode("utf-8")) > room:
        text = text.encode("utf-8")[: max(0, room)].decode("utf-8", "ignore").rstrip()
    description = f"{text}\n\n{tag_line}" if text else tag_line

    tags: list[str] = []
    used = 0
    for t in [h.lstrip("#") for h in hashtags] + [p.game_name]:
        if t.lower() in {x.lower() for x in tags}:
            continue
        if used + len(t) + 1 > TAGS_MAX_CHARS:
            break
        tags.append(t)
        used += len(t) + 1

    snippet = {
        "title": build_title(sidecar, cfg),
        "description": description,
        "tags": tags,
        "categoryId": str(p.get("category_id", "20")),
    }
    lang = p.get("default_language")
    if lang:
        snippet["defaultLanguage"] = lang
        snippet["defaultAudioLanguage"] = lang

    status: dict[str, Any] = {"selfDeclaredMadeForKids": False}
    if publish_at is not None:
        status["privacyStatus"] = "private"        # required for publishAt
        status["publishAt"] = publish_at if isinstance(publish_at, str) else to_rfc3339(publish_at)
    else:
        status["privacyStatus"] = privacy
    return {"snippet": snippet, "status": status}


# --------------------------------------------------------------------- errors

def _http_reasons(err: Any) -> set[str]:
    reasons: set[str] = set()
    for d in getattr(err, "error_details", None) or []:
        if isinstance(d, dict) and d.get("reason"):
            reasons.add(str(d["reason"]))
    content = getattr(err, "content", None)
    if content:
        try:
            data = json.loads(content.decode("utf-8") if isinstance(content, bytes) else content)
            e = data.get("error", {}) if isinstance(data, dict) else {}
            for x in e.get("errors", []) or []:
                if x.get("reason"):
                    reasons.add(str(x["reason"]))
            st = e.get("status")
            if st:
                reasons.add(str(st))
        except Exception:
            pass
    return reasons


def _http_status(err: Any) -> int:
    resp = getattr(err, "resp", None)
    try:
        return int(getattr(resp, "status", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _connection_errors() -> tuple[type[BaseException], ...]:
    errs: list[type[BaseException]] = [ConnectionError, TimeoutError, socket.timeout]
    try:
        import httplib2
        errs.append(httplib2.HttpLib2Error)
    except Exception:  # pragma: no cover
        pass
    try:
        import http.client
        errs.append(http.client.HTTPException)
    except Exception:  # pragma: no cover
        pass
    return tuple(errs)


# --------------------------------------------------------------------- upload

def upload(youtube: Any, item: QueueItem, sidecar: ClipSidecar, cfg: Any,
           publish_at: datetime | str | None = None, privacy: str = "private",
           sleep: Callable[[float], None] | None = None,
           on_tick: Callable[[], None] | None = None) -> str:
    """Upload one clip. Returns the new video id.

    Retries 5xx and connection errors with exponential backoff (max MAX_RETRIES). Raises
    QuotaExceeded on 403/429 quota or rate errors, UploadError on anything else. The PAUSE file
    is re-checked before every chunk; if it appears, raises UploadPaused (clean abort).
    """
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    sleep = sleep or _sleep
    body = build_body(item, sidecar, cfg, publish_at=publish_at, privacy=privacy)
    path = item.clip_path or sidecar.clip_path
    try:
        media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=CHUNK_SIZE, resumable=True)
    except (OSError, IOError) as e:
        raise UploadError(f"clip file unreadable: {path}: {e}") from e

    conn_errs = _connection_errors()
    retries = 0
    try:
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            if _paused(cfg):
                raise UploadPaused(f"PAUSE appeared during upload of {item.clip_id}")
            try:
                _status, response = request.next_chunk()
                if on_tick:
                    on_tick()
            except HttpError as e:
                code = _http_status(e)
                reasons = _http_reasons(e)
                if code in (403, 429) and (reasons & QUOTA_REASONS or code == 429):
                    raise QuotaExceeded(f"HTTP {code} {sorted(reasons)}") from e
                if code not in RETRY_STATUS:
                    raise UploadError(f"HTTP {code} {sorted(reasons)}") from e
                err = f"HTTP {code}"
            except conn_errs as e:
                err = f"{type(e).__name__}: {e}"
            else:
                continue
            retries += 1
            if retries > MAX_RETRIES:
                raise UploadError(f"gave up after {MAX_RETRIES} retries: {err}")
            wait = min(60.0, 2.0 ** retries) * (0.5 + random.random() / 2)
            log.warning("upload %s: %s, retry %d/%d in %.1fs", item.clip_id, err, retries,
                        MAX_RETRIES, wait)
            _sleep_ticking(sleep, wait, on_tick)
    finally:
        fd = getattr(media, "_fd", None)
        try:
            if fd is not None:
                fd.close()
        except Exception:
            pass
    vid = response.get("id") if isinstance(response, dict) else None
    if not vid:
        raise UploadError(f"upload finished without a video id: {str(response)[:200]}")
    return vid


def _paused(cfg: Any) -> bool:
    try:
        from .. import config
        return bool(config.paused(cfg))
    except Exception:
        return False


def _sleep_ticking(sleep: Callable[[float], None], seconds: float,
                   on_tick: Callable[[], None] | None) -> None:
    """Sleep in <=5 s steps, ticking the heartbeat between steps."""
    left = seconds
    while left > 0:
        step = min(5.0, left)
        sleep(step)
        left -= step
        if on_tick:
            on_tick()
