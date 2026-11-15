"""Gameplay track signals: audio spike, on-screen action, HUD events, sound type, duration."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import AudioFeatures, VisualFeatures
from .chat import chat_events
from .candidates import (Activity, amax, bursts, clip01, dominant_scene_tag, in_ranges, motion_slice, p90,
                         radio_factor, radio_share, sget, window_frames)

GAMEPLAY_SIGNALS = ("audio_spike", "action", "hud_events", "sound_type", "duration", "chat")
AI_ONLY = ("hud_events",)
RADIO_MASKED = ("audio_spike", "sound_type")

HUD_TEXT = {"wasted": "WASTED", "busted": "BUSTED", "mission_passed": "mission passed",
            "mission_failed": "mission failed"}


@dataclass
class SignalResult:
    raw: dict[str, float]                       # measured signals (no radio masking), 0..1
    scored: dict[str, float]                    # what the score uses (radio-masked), 0..1
    available: set[str]                         # signals that could be measured
    events: list[str] = field(default_factory=list)
    scene_tag: str | None = None


def _sound_type(tr: np.ndarray, burst_full: float) -> float:
    if tr.size == 0:
        return 0.0
    return clip01(0.5 * amax(tr) + 0.5 * min(1.0, bursts(tr) / burst_full))


def cv_action(visual: VisualFeatures, ws: float, we: float) -> tuple[float, dict]:
    m = visual.motion
    ms = motion_slice(visual, ws, we)
    if not ms.any():
        return 0.0, {"motion": 0.0, "cuts": 0, "shake": 0.0, "flash": 0.0}
    mot = p90(np.asarray(m.motion, float)[ms])
    cuts = bursts(np.asarray(m.scene_change, float)[ms])
    shake = p90(np.asarray(m.shake, float)[ms])
    flash = amax(np.asarray(m.flash, float)[ms])
    v = clip01(0.4 * mot + 0.2 * min(1.0, cuts / 3.0) + 0.2 * shake + 0.2 * flash)
    return v, {"motion": mot, "cuts": cuts, "shake": shake, "flash": flash}


def hud_signal(visual: VisualFeatures, ws: float, we: float, peak: float) -> tuple[float, list[str]]:
    frames = window_frames(visual, ws, we, peak)
    if not frames:
        return 0.0, []
    before = [f for f in visual.frames if f.t < ws]
    prev_stars = before[-1].wanted_stars if before else frames[0].wanted_stars
    stars = max(int(f.wanted_stars or 0) for f in frames)
    evs = {e for f in frames for e in (f.hud_events or ())}
    rise = stars > prev_stars or "wanted_up" in evs
    v = 0.0
    events: list[str] = []
    for key in ("wasted", "busted", "mission_passed"):
        if key in evs:
            v = 1.0
            events.append(HUD_TEXT[key])
    if "mission_failed" in evs:
        v = max(v, 0.7)
        events.append(HUD_TEXT["mission_failed"])
    if rise and stars > 0:
        v = max(v, min(1.0, 0.4 + 0.12 * stars))
        events.append(f"wanted level up to {stars} stars")
    elif rise:
        v = max(v, 0.5)
        events.append("wanted level up")
    elif stars > 0:
        v = max(v, 0.2 * stars / 5.0)
        events.append(f"{stars} wanted stars")
    if evs & {"notification", "other"}:
        v = max(v, 0.1)
    return clip01(v), events


def gameplay_signals(ws: float, we: float, peak: float, audio: AudioFeatures, visual: VisualFeatures,
                     act: Activity, cfg, chat=None) -> SignalResult:
    ai_ok = bool(visual.ai_ok)
    fac = radio_factor(cfg)
    burst_full = float(sget(cfg, "burst_full", 6))
    raw: dict[str, float] = {}
    scored: dict[str, float] = {}
    events: list[str] = []

    sl = audio.at(ws, we)
    at = np.asarray(audio.t, float)[sl]
    mask = np.where(in_ranges(at, audio.radio_ranges), fac, 1.0)
    spike = np.asarray(audio.spike, float)[sl]
    tr = np.asarray(audio.transient, float)[sl]
    raw["audio_spike"] = clip01(amax(spike))
    scored["audio_spike"] = clip01(amax(spike * mask))
    raw["sound_type"] = _sound_type(tr, burst_full)
    scored["sound_type"] = _sound_type(tr * mask, burst_full)

    cv, parts = cv_action(visual, ws, we)
    frames = window_frames(visual, ws, we, peak)
    if ai_ok and frames:
        acts = sorted((int(f.action or 0) for f in frames), reverse=True)[:2]
        ai_act = clip01(sum(acts) / len(acts) / 10.0)
        b = float(sget(cfg, "ai_action_blend", 0.5))
        action = clip01((1 - b) * cv + b * ai_act)
        if max(acts) >= 7:
            events.append(f"high on-screen action ({max(acts)}/10)")
    else:
        action = cv
    raw["action"] = scored["action"] = action

    available = {"audio_spike", "action", "sound_type", "duration"}
    if ai_ok:
        hud, hev = hud_signal(visual, ws, we, peak)
        raw["hud_events"] = scored["hud_events"] = hud
        events = hev + events
        available.add("hud_events")

    # duration: seconds of sustained activity inside the window (one bang is not enough)
    lo, hi = max(0, int(np.floor(ws))), min(act.n, int(np.ceil(we)))
    level = float(sget(cfg, "sustain_level", 0.35))
    sustained = int(np.count_nonzero(act.total[lo:hi] >= level)) if hi > lo else 0
    raw["duration"] = scored["duration"] = clip01(sustained / float(sget(cfg, "sustain_full_s", 8)))

    if chat is not None and chat.value is not None:      # ChatSignal; None/unavailable -> dropped
        raw["chat"] = scored["chat"] = float(chat.value)
        available.add("chat")
        events = events + chat_events(chat)

    if raw["audio_spike"] >= 0.6:
        events.append("big audio spike")
    if raw["sound_type"] >= 0.5:
        events.append("gunfire / explosion bursts")
    if parts["flash"] >= 0.6:
        events.append("flash on screen")
    if parts["shake"] >= 0.5:
        events.append("screen shake")
    if sustained >= 5:
        events.append(f"sustained action {sustained} s")
    if radio_share(audio, ws, we) > 0.3:
        events.append("radio playing")

    tag = dominant_scene_tag(frames) if ai_ok else None
    return SignalResult(raw=raw, scored=scored, available=available, events=events, scene_tag=tag)
