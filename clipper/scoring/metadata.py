"""Clip metadata (title, description, hashtags, hook) from the judge, or a template from the events.

Rules: the title always contains cfg.publish.game_name and is <= 100 chars; hashtags always
include #Shorts and cfg.publish.base_hashtags (deduplicated, max 8); cutscene clips are spoilers
and get "spoiler" in the title; the hook is <= 40 chars.
"""
from __future__ import annotations

import re

from ..models import Candidate, ClipMeta

TITLE_MAX = 100
HOOK_MAX = 40
HASHTAGS_MAX = 8

_BAD_TITLE_CHARS = re.compile(r"[<>\r\n\t]")


def _cut(s: str, n: int) -> str:
    s = " ".join(str(s or "").split())
    if len(s) <= n:
        return s
    cut = s[: n - 1]
    if " " in cut[n // 2:]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:-") + "…"


def make_title(base: str, game: str, spoiler: bool, max_len: int = TITLE_MAX) -> str:
    base = _BAD_TITLE_CHARS.sub(" ", str(base or "")).strip()
    base = " ".join(base.split()) or f"Crazy moment in {game}"
    prefix = "[Spoiler] " if spoiler and "spoiler" not in base.lower() else ""
    suffix = "" if game.lower() in base.lower() else f" ({game})"
    room = max_len - len(prefix) - len(suffix)
    if len(base) > room:
        base = _cut(base, room)
        if not suffix and game.lower() not in base.lower():   # truncation removed the game name
            suffix = f" ({game})"
            base = _cut(base, max_len - len(prefix) - len(suffix))
    return (prefix + base + suffix)[:max_len]


def make_hook(text: str, max_len: int = HOOK_MAX) -> str:
    return _cut(text, max_len)


def _norm_tag(tag: str) -> str:
    t = re.sub(r"[^\w]", "", str(tag or "").strip().lstrip("#"))
    return f"#{t}" if t else ""


def make_hashtags(extra: list[str] | None, base: list[str] | None, max_n: int = HASHTAGS_MAX) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in ["#Shorts", *(base or []), *(extra or [])]:
        n = _norm_tag(t)
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out[:max_n]


# --------------------------------------------------------------------- templates
def _has(events: list[str], *needles: str) -> bool:
    ev = " | ".join(events).lower()
    return any(n.lower() in ev for n in needles)


def _stars(events: list[str]) -> tuple[int, bool]:
    for e in events:
        m = re.search(r"wanted level up to (\d) stars", e)
        if m:
            return int(m.group(1)), True
    for e in events:
        m = re.search(r"(\d) wanted stars", e)
        if m:
            return int(m.group(1)), False
    return 0, _has(events, "wanted level up")


def template_texts(c: Candidate, game: str) -> tuple[str, str, str, list[str]]:
    """(title, hook, description, extra hashtags) for a candidate without a judge answer."""
    ev = c.events or []
    dur = int(round(c.duration_s))
    if c.track == "cutscene":
        m = next((re.search(r'title card: "(.+)"', e) for e in ev if e.startswith("title card")), None)
        if m:
            text = m.group(1).strip()
            title, hook = f"{text.title()} cutscene in {game}", text
        elif c.scene_tag and _has(ev, "new scene"):
            title, hook = f"New {c.scene_tag} scene in {game}", f"First look: {c.scene_tag}"
        elif _has(ev, "big spectacle"):
            title, hook = f"Epic cutscene moment in {game}", "This looks unreal"
        else:
            title, hook = f"Story cutscene from {game}", "Story moment"
        desc = f"A cutscene moment from a live {game} stream. Contains story spoilers."
        return title, hook, desc, ["#GTA6Story", "#cutscene", "#gaming"]

    stars, rising = _stars(ev)
    if _has(ev, "mission passed"):
        title, hook = f"Mission passed! ({game})", "Mission passed!"
    elif _has(ev, "WASTED"):
        title, hook = f"Wasted in {game}", "WASTED"
    elif _has(ev, "BUSTED"):
        title, hook = f"Busted in {game}", "BUSTED"
    elif stars >= 3:
        title = f"{stars}-star chase in {game}"
        hook = f"{stars} stars in {dur} seconds" if rising else f"{stars} stars and counting"
    elif rising:
        title, hook = f"Cops on my tail in {game}", "Wanted level rising"
    elif _has(ev, "mission failed"):
        title, hook = f"Mission failed... ({game})", "That did not go well"
    elif _has(ev, "flash on screen") or (_has(ev, "big audio spike") and _has(ev, "screen shake")):
        title, hook = f"Huge explosion in {game}", "That escalated fast"
    elif _has(ev, "gunfire"):
        title, hook = f"Wild shootout in {game}", "Bullets everywhere"
    elif _has(ev, "sustained action"):
        title, hook = f"Non-stop action in {game}", "No time to breathe"
    else:
        title, hook = f"Crazy moment in {game}", "Wait for it..."
    desc = f"{title}. Clipped live from a {game} stream."
    return title, hook, desc, ["#gaming", "#GTA", "#GTAVI"]


def build_meta(c: Candidate, cfg) -> ClipMeta:
    pub = cfg.publish
    game = str(pub.game_name)
    base_tags = list(pub.get("base_hashtags") or [])
    j = c.judge if isinstance(c.judge, dict) else None
    t_title, t_hook, t_desc, t_tags = template_texts(c, game)
    spoiler = c.track == "cutscene" or bool(j and j.get("spoiler"))
    if j and str(j.get("title", "")).strip():
        title = make_title(j["title"], game, spoiler)
        hook = make_hook(j.get("hook") or t_hook)
        desc = str(j.get("description") or t_desc).strip()[:4500]
        tags = make_hashtags(list(j.get("hashtags") or []), base_tags)
        source = "judge"
    else:
        title = make_title(t_title, game, spoiler)
        hook = make_hook(t_hook)
        desc = t_desc
        tags = make_hashtags(t_tags, base_tags)
        source = "template"
    return ClipMeta(title=title, description=desc, hashtags=tags, hook=hook, spoiler=spoiler, source=source)
