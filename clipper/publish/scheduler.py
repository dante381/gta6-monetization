"""Publish planning: slots, daily caps, rollover and the Pacific-day upload counter.

Pure functions. Every function takes `now` (an aware datetime) so tests can pin the clock.

Time zones:
- Caps count Shorts per *publish day* in local time. "Local" is `cfg.publish.timezone`:
  "" means the laptop's own time zone, "UTC" means UTC, anything else is an IANA name
  (falls back to the laptop's zone if tzdata is missing).
- The API upload limit resets at midnight Pacific time. Windows Python often has no tz
  database (`tzdata` is not installed in this venv), so when `zoneinfo` can't find
  America/Los_Angeles we compute US Pacific time ourselves: UTC-8, or UTC-7 while US daylight
  saving is on (second Sunday of March 02:00 PST to first Sunday of November 02:00 PDT).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone, tzinfo
from typing import Any, Iterable, Sequence

from ..models import QueueItem

# --------------------------------------------------------------------- time helpers

try:  # pragma: no cover - depends on the machine
    from zoneinfo import ZoneInfo
    _PACIFIC: tzinfo | None = ZoneInfo("America/Los_Angeles")
except Exception:  # ZoneInfoNotFoundError on Windows without tzdata
    _PACIFIC = None


def _nth_sunday(year: int, month: int, n: int) -> date:
    d = date(year, month, 1)
    first_sunday = d + timedelta(days=(6 - d.weekday()) % 7)
    return first_sunday + timedelta(weeks=n - 1)


def _pacific_offset(dt_utc: datetime) -> timedelta:
    """Fixed-rule US Pacific offset for an aware datetime (fallback when tzdata is missing)."""
    y = dt_utc.year
    dst_start = datetime.combine(_nth_sunday(y, 3, 2), dtime(10, 0), timezone.utc)   # 02:00 PST
    dst_end = datetime.combine(_nth_sunday(y, 11, 1), dtime(9, 0), timezone.utc)     # 02:00 PDT
    return timedelta(hours=-7) if dst_start <= dt_utc < dst_end else timedelta(hours=-8)


def to_pacific(dt: datetime) -> datetime:
    dt = ensure_aware(dt)
    if _PACIFIC is not None:
        return dt.astimezone(_PACIFIC)
    u = dt.astimezone(timezone.utc)
    return u.astimezone(timezone(_pacific_offset(u)))


def pacific_day(dt: datetime) -> str:
    """YYYY-MM-DD of `dt` in US Pacific time (the YouTube upload-limit day)."""
    return to_pacific(dt).date().isoformat()


def next_pacific_midnight(dt: datetime) -> datetime:
    """The UTC instant when the next Pacific day starts (upload limits reset)."""
    p = to_pacific(dt)
    nxt = p.date() + timedelta(days=1)
    if _PACIFIC is not None:
        return datetime.combine(nxt, dtime(0), _PACIFIC).astimezone(timezone.utc)
    guess = datetime.combine(nxt, dtime(8, 0), timezone.utc)          # midnight PST
    off = _pacific_offset(guess - timedelta(hours=1))
    return datetime.combine(nxt, dtime(0), timezone(off)).astimezone(timezone.utc)


def ensure_aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_rfc3339(dt: datetime) -> str:
    """RFC3339 UTC with a 'Z' suffix, as the YouTube API wants for publishAt."""
    return ensure_aware(dt).astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_rfc3339(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return ensure_aware(datetime.fromisoformat(s))


def local_tz(cfg: Any) -> tzinfo | None:
    """tzinfo for cap counting; None means the laptop's own zone (via dt.astimezone())."""
    name = ""
    try:
        name = (cfg.publish.get("timezone") or "").strip()
    except AttributeError:
        pass
    if not name or name.lower() == "local":
        return None
    if name.upper() in ("UTC", "Z", "GMT"):
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return None


def _to_local(dt: datetime, tz: tzinfo | None) -> datetime:
    dt = ensure_aware(dt)
    return dt.astimezone(tz) if tz is not None else dt.astimezone()


def local_date(dt: datetime, tz: tzinfo | None) -> date:
    return _to_local(dt, tz).date()


def next_local_midnight(dt: datetime, tz: tzinfo | None) -> datetime:
    nxt = local_date(dt, tz) + timedelta(days=1)
    if tz is not None:
        return datetime.combine(nxt, dtime(0), tz).astimezone(timezone.utc)
    return datetime.combine(nxt, dtime(0)).astimezone().astimezone(timezone.utc)


# --------------------------------------------------------------------- planning

@dataclass
class PlannedUpload:
    item: QueueItem
    kind: str                       # publish | backup
    privacy: str                    # private (scheduled) | unlisted (backup)
    publish_at: datetime | None     # UTC slot for scheduled posts

    @property
    def publish_at_str(self) -> str | None:
        return to_rfc3339(self.publish_at) if self.publish_at else None


def _sort_best_first(items: Iterable[QueueItem]) -> list[QueueItem]:
    return sorted(items, key=lambda q: (-(q.final_score or 0.0), q.created_at or 0.0, q.clip_id))


def assign_slots(pending: Sequence[QueueItem], existing: Sequence[QueueItem], now: datetime,
                 slot_hours: float, first_delay_min: float, daily_total: int, daily_cutscene: int,
                 tz: tzinfo | None = None) -> list[tuple[QueueItem, datetime]]:
    """Give every pending publish item a publish slot (UTC), best score first.

    - Slots are `slot_hours` apart, the first no sooner than now + `first_delay_min` and one
      slot after the latest already-scheduled publish_at.
    - At most `daily_total` Shorts and `daily_cutscene` cutscene Shorts per local publish day,
      counting the already-scheduled items by their publish_at date.
    - When a day is full, the rest roll to the next day's slots, highest score first. A
      cutscene that can't fit today waits while gameplay clips fill today's slots.
    """
    now = ensure_aware(now)
    slot = timedelta(hours=float(slot_hours))
    if slot <= timedelta(0):
        raise ValueError("slot_hours must be > 0")
    total: dict[date, int] = {}
    cut: dict[date, int] = {}
    latest: datetime | None = None
    for q in existing:
        if not q.publish_at:
            continue
        t = parse_rfc3339(q.publish_at)
        d = local_date(t, tz)
        total[d] = total.get(d, 0) + 1
        if q.track == "cutscene":
            cut[d] = cut.get(d, 0) + 1
        latest = t if latest is None or t > latest else latest

    t = now + timedelta(minutes=float(first_delay_min))
    if latest is not None and latest + slot > t:
        t = latest + slot
    t = t.replace(microsecond=0)

    remaining = _sort_best_first(pending)
    out: list[tuple[QueueItem, datetime]] = []
    if daily_total <= 0:
        return out
    guard = 0
    while remaining:
        guard += 1
        if guard > 100_000:   # can't happen with sane caps; never loop forever
            break
        d = local_date(t, tz)
        if total.get(d, 0) >= daily_total:
            t = max(t, next_local_midnight(t, tz))
            continue
        cut_ok = cut.get(d, 0) < daily_cutscene
        pick = next((q for q in remaining if q.track != "cutscene" or cut_ok), None)
        if pick is None:      # only cutscenes left and today's cutscene cap is used up
            if daily_cutscene <= 0:
                break
            t = max(t, next_local_midnight(t, tz))
            continue
        remaining.remove(pick)
        out.append((pick, t))
        total[d] = total.get(d, 0) + 1
        if pick.track == "cutscene":
            cut[d] = cut.get(d, 0) + 1
        t = t + slot
    return out


def upload_budget(limit: int, uploads_today: int) -> int:
    return max(0, int(limit) - int(uploads_today))


def plan_uploads(pending_publish: Sequence[QueueItem], pending_backup: Sequence[QueueItem],
                 existing: Sequence[QueueItem], now: datetime, cfg: Any, uploads_today: int,
                 tz: tzinfo | None = None) -> list[PlannedUpload]:
    """What to upload this tick, in order: scheduled publish items (best first), then backups
    (unlisted, best first) if `cfg.publish.upload_backups`, all within the Pacific-day limit."""
    p = cfg.publish
    caps = cfg.locked.caps
    budget = upload_budget(p.get("daily_upload_limit", 100), uploads_today)
    slots = assign_slots(pending_publish, existing, now, p.slot_hours, p.first_slot_delay_min,
                         int(caps.daily_total), int(caps.daily_cutscene), tz)
    out: list[PlannedUpload] = []
    for item, when in slots:
        if len(out) >= budget:
            return out
        out.append(PlannedUpload(item, "publish", "private", when))
    if p.get("upload_backups", True):
        for item in _sort_best_first(pending_backup):
            if len(out) >= budget:
                break
            out.append(PlannedUpload(item, "backup", "unlisted", None))
    return out
