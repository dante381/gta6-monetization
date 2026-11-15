"""Cutscene track: window trimming and signals (title text, spectacle, newness, music swell)."""
from __future__ import annotations

import numpy as np

from ..models import AudioFeatures, VisualFeatures
from .candidates import (Activity, amax, bursts, clip01, dominant_scene_tag, motion_slice, p90, sget,
                         window_frames)
from .gameplay import SignalResult

CUTSCENE_SIGNALS = ("title_text", "spectacle", "newness", "music_swell")
AI_ONLY = ("title_text", "newness")


def _cut_region(ind: np.ndarray, start: int, max_gap: int = 3) -> tuple[int, int]:
    """Contiguous run of True around `start` (gaps up to max_gap seconds bridged) -> [r0, r1)."""
    n = len(ind)
    r0 = r1 = start
    gap = 0
    i = start
    while i - 1 >= 0:
        i -= 1
        if ind[i]:
            r0, gap = i, 0
        else:
            gap += 1
            if gap > max_gap:
                break
    gap = 0
    i = start
    while i + 1 < n:
        i += 1
        if ind[i]:
            r1, gap = i, 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return r0, r1 + 1


def cutscene_window(act: Activity, peak: int, ws: float, we: float, cfg,
                    chunk_start_s: float = 0.0) -> tuple[float, float]:
    """Cover the cutscene around the peak; long cutscenes -> strongest cutscene_min_s..clip_max_s piece."""
    cmin = float(sget(cfg, "cutscene_min_s", 30))
    cmax = float(sget(cfg, "clip_max_s", 45))
    ind = act.cut_indicator()
    lo, hi = max(0, int(np.floor(ws))), min(act.n, int(np.ceil(we)))
    cand = [i for i in range(lo, hi) if ind[i]]
    if not cand:
        return ws, min(we, ws + cmax)
    seed = min(cand, key=lambda i: abs(i - peak))
    r0, r1 = _cut_region(ind, seed)
    L = r1 - r0
    if L <= cmax:
        if L >= cmin:
            s, e = float(r0), float(r1)
        else:
            s = r0 - (cmin - L) / 2.0
            e = s + cmin
    else:
        W = int(cmax)
        csum = np.concatenate([[0.0], np.cumsum(act.total)])
        lo_s = max(r0, peak - W + 1)
        hi_s = min(peak, r1 - W)
        if lo_s > hi_s:                       # peak near a region edge
            lo_s, hi_s = r0, r1 - W
        starts = np.arange(lo_s, hi_s + 1)
        sums = csum[starts + W] - csum[starts]
        s0 = int(starts[int(np.argmax(sums))])
        s, e = s0, s0 + W
        mean = float(sums.max()) / W
        while e - s > cmin and e - 1 > peak and act.total[e - 1] < 0.5 * mean:
            e -= 1
        while e - s > cmin and s < peak and act.total[s] < 0.5 * mean:
            s += 1
        s, e = float(s), float(e)
    if chunk_start_s + s < 0:
        e += -(chunk_start_s + s)
        s = -chunk_start_s
    return s, e


def cutscene_signals(ws: float, we: float, peak: float, audio: AudioFeatures, visual: VisualFeatures,
                     act: Activity, cfg, state) -> SignalResult:
    frames = window_frames(visual, ws, we, peak)
    ai_ok = bool(visual.ai_ok)
    events: list[str] = ["letterbox cutscene"]
    raw: dict[str, float] = {}

    # spectacle: motion, cuts, flashes, loudness
    m = visual.motion
    ms = motion_slice(visual, ws, we)
    mot = np.asarray(m.motion, float)[ms] if ms.any() else np.zeros(0)
    cuts = bursts(np.asarray(m.scene_change, float)[ms]) if ms.any() else 0
    flash = np.asarray(m.flash, float)[ms] if ms.any() else np.zeros(0)
    sl = audio.at(ws, we)
    loud = clip01(amax(np.asarray(audio.loud_z, float)[sl]) / 3.0)
    raw["spectacle"] = clip01(0.35 * p90(mot) + 0.25 * min(1.0, cuts / 4.0) + 0.2 * amax(flash) + 0.2 * loud)
    if raw["spectacle"] >= 0.6:
        events.append("big spectacle")

    raw["music_swell"] = clip01(amax(np.asarray(audio.swell, float)[sl]))
    if raw["music_swell"] >= 0.5:
        events.append("music swell")

    tag = dominant_scene_tag(frames) if ai_ok else None
    available = {"spectacle", "music_swell"}
    if ai_ok:
        available |= {"title_text", "newness"}
        titles = [str(f.title_text).strip() for f in frames if f.title_text and str(f.title_text).strip()]
        raw["title_text"] = 1.0 if titles else 0.0
        if titles:
            events.append(f'title card: "{titles[0][:60]}"')
        if tag:
            count = state.scene_count(tag) if state is not None else 0
            raw["newness"] = clip01(float(sget(cfg, "newness_decay", 0.5)) ** count)
            if count == 0:
                events.append(f"new scene: {tag}")
        else:
            raw["newness"] = 0.5
    return SignalResult(raw=raw, scored=dict(raw), available=available, events=events, scene_tag=tag)
