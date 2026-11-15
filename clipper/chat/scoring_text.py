"""Text heuristics for chat: hype score, explicit clip requests, sanitization for stored samples."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

HYPE_CAP = 2.0
_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|io|tv|gg|co|me|ly|be)\b/?\S*")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_MENTION = re.compile(r"@\S+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{5,}\d")
_ELONG = re.compile(r"([A-Za-z])\1{2,}")
_WS = re.compile(r"\s+")


def _opt(cfg: Any, key: str, default: Any) -> Any:
    try:
        v = cfg.get(key, default)
    except AttributeError:
        v = getattr(cfg, key, default)
    return default if v is None else v


@lru_cache(maxsize=16)
def _word_re(words: tuple[str, ...]) -> re.Pattern | None:
    ws = sorted({w.strip().lower() for w in words if w and w.strip()}, key=len, reverse=True)
    if not ws:
        return None
    return re.compile(r"(?<![\w])(?:" + "|".join(re.escape(w) for w in ws) + r")(?![\w])", re.IGNORECASE)


def hype_score(text: str, cfg: Any) -> float:
    """0..2 per message: hype words (word boundaries), emotes, elongated letters, ALL-CAPS."""
    if not text:
        return 0.0
    score = 0.0
    rx = _word_re(tuple(_opt(cfg, "hype_words", ())))
    if rx is not None:
        hits = {m.group(0).lower() for m in rx.finditer(text)}
        if _ELONG.search(text):  # "LMAOOOO" should still count as the word "lmao"
            hits |= {m.group(0).lower() for m in rx.finditer(_ELONG.sub(r"\1", text))}
        score += min(len(hits), 2) * 0.8
    emotes = _opt(cfg, "hype_emotes", ())
    score += min(sum(text.count(e) for e in emotes), 3) * 0.4
    if _ELONG.search(text):
        score += 0.5
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        score += 0.5
    return min(score, HYPE_CAP)


def clip_request(text: str, cfg: Any) -> bool:
    low = (text or "").lower()
    return any(p and p.lower() in low for p in _opt(cfg, "clip_phrases", ()))


def sanitize(text: str, cfg: Any) -> str:
    """Strip @mentions, URLs, emails, phone-like digit runs; collapse space; truncate."""
    if not text:
        return ""
    t = _EMAIL.sub(" ", text)
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _PHONE.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    n = int(_opt(cfg, "sample_max_chars", 80))
    return t[:n].rstrip()
