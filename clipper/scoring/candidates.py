"""Per-second activity curve, peak picking, candidate windows and track assignment.

All times in this module are chunk-relative seconds unless a name says `stream`.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..models import AudioFeatures, Chunk, VisualFeatures, VisualFrame

MENU_KINDS = ("menu", "loading", "map")
BIG_HUD = ("wasted", "busted", "mission_passed")

DEFAULT_ACTIVITY_WEIGHTS = {"audio": 0.4, "motion": 0.3, "ai": 0.3, "letterbox": 0.3}


# --------------------------------------------------------------------- config helpers
def sget(cfg, key: str, default):
    """cfg.scoring.<key> with a default (optional keys may be missing in older settings)."""
    sc = cfg.get("scoring", {}) or {}
    v = sc.get(key, default)
    return default if v is None else v


def radio_factor(cfg) -> float:
    return float((sget(cfg, "penalties", {}) or {}).get("radio_audio_factor", 0.3))


# --------------------------------------------------------------------- small numeric helpers
def chunk_duration(chunk: Chunk, audio: AudioFeatures | None, visual: VisualFeatures | None) -> float:
    d = float(getattr(chunk, "duration_s", 0.0) or 0.0)
    if d > 0:
        return d
    ends = [0.0]
    if audio is not None and len(audio.t):
        ends.append(float(audio.t[-1]) + float(audio.hop_s) / 2)
    if visual is not None and visual.motion is not None and len(visual.motion.t):
        ends.append(float(visual.motion.t[-1]) + 0.5)
    if visual is not None and visual.frames:
        ends.append(max(f.t for f in visual.frames) + 3.0)
    return max(ends)


def in_ranges(t: np.ndarray, ranges: Iterable[tuple[float, float]]) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    m = np.zeros(t.shape, dtype=bool)
    for a, b in ranges or ():
        m |= (t >= a) & (t < b)
    return m


def per_second(t, v, n: int, how: str = "max") -> np.ndarray:
    """Pool a series sampled at times t into n one-second bins [i, i+1); empty bins interpolated."""
    out = np.zeros(n, dtype=float)
    if n <= 0 or t is None or len(t) == 0:
        return out
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    idx = np.floor(t).astype(int)
    ok = (idx >= 0) & (idx < n)
    idx, v = idx[ok], v[ok]
    if idx.size == 0:
        return out
    cnt = np.bincount(idx, minlength=n)
    if how == "max":
        acc = np.full(n, -np.inf)
        np.maximum.at(acc, idx, v)
    else:
        acc = np.bincount(idx, weights=v, minlength=n) / np.maximum(cnt, 1)
    have = cnt > 0
    xs = np.arange(n)
    out = np.interp(xs, xs[have], acc[have])
    return out


def p90(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, 90)) if x.size else 0.0


def amax(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.max()) if x.size else 0.0


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def bursts(x: np.ndarray, thr: float = 0.5) -> int:
    """Number of rising edges above thr (separate sharp bursts)."""
    x = np.asarray(x, dtype=float) >= thr
    if x.size == 0:
        return 0
    return int(x[0]) + int(np.count_nonzero(x[1:] & ~x[:-1]))


def weighted_score(values: dict[str, float], weights: dict, available: Iterable[str] | None = None) -> float:
    """100 * sum(w*v) / sum(w) over the measurable signals.

    When some signals can't be measured (Claude unavailable) the remaining weights are
    rescaled so the best possible score is still 100 (spec: fall back to audio + motion).
    """
    keys = [k for k in weights if available is None or k in available]
    wsum = sum(float(weights[k]) for k in keys)
    if wsum <= 0:
        return 0.0
    return 100.0 * sum(float(weights[k]) * float(values.get(k, 0.0)) for k in keys) / wsum


# --------------------------------------------------------------------- window helpers
def window_frames(visual: VisualFeatures, ws: float, we: float, peak: float | None = None) -> list[VisualFrame]:
    fr = [f for f in visual.frames if ws <= f.t < we]
    if not fr and visual.frames and peak is not None:
        fr = [min(visual.frames, key=lambda f: abs(f.t - peak))]
    return fr


def motion_slice(visual: VisualFeatures, ws: float, we: float) -> np.ndarray:
    t = np.asarray(visual.motion.t, dtype=float)
    return (t >= ws) & (t < we)


def letterbox_share(visual: VisualFeatures, ws: float, we: float) -> float:
    shares = []
    fr = [f for f in visual.frames if ws <= f.t < we]
    if fr:
        shares.append(sum(1 for f in fr if f.letterbox) / len(fr))
    m = motion_slice(visual, ws, we)
    if m.any():
        shares.append(float(np.asarray(visual.motion.letterbox, dtype=bool)[m].mean()))
    return max(shares) if shares else 0.0


def ai_cutscene_majority(visual: VisualFeatures, ws: float, we: float) -> bool:
    fr = [f for f in visual.frames if ws <= f.t < we and f.from_ai]
    if not fr:
        return False
    return sum(1 for f in fr if f.kind == "cutscene") > len(fr) / 2


def assign_track(visual: VisualFeatures, ws: float, we: float, cfg) -> str:
    if letterbox_share(visual, ws, we) >= float(sget(cfg, "letterbox_share", 0.6)):
        return "cutscene"
    if visual.ai_ok and ai_cutscene_majority(visual, ws, we):
        return "cutscene"
    return "gameplay"


def dominant_scene_tag(frames: list[VisualFrame]) -> str | None:
    tags = [f.scene_tag.strip().lower() for f in frames if f.scene_tag and f.scene_tag.strip()]
    if not tags:
        return None
    return Counter(tags).most_common(1)[0][0]


def radio_ranges_in(audio: AudioFeatures, ws: float, we: float) -> list[tuple[float, float]]:
    out = []
    for a, b in audio.radio_ranges or ():
        a2, b2 = max(a, ws), min(b, we)
        if b2 > a2:
            out.append((float(a2), float(b2)))
    return out


def radio_share(audio: AudioFeatures, ws: float, we: float) -> float:
    if we <= ws:
        return 0.0
    return sum(b - a for a, b in radio_ranges_in(audio, ws, we)) / (we - ws)


# --------------------------------------------------------------------- activity curve
@dataclass
class Activity:
    n: int                        # number of one-second bins (chunk duration, rounded up)
    duration: float
    audio: np.ndarray             # radio-masked audio component (spike/transient)
    audio_raw: np.ndarray
    cv: np.ndarray                # motion + cuts + flashes + shake
    ai: np.ndarray                # Claude action / HUD events / title text
    letterbox: np.ndarray         # 0..1 share of the second with bars
    cut_ai: np.ndarray            # bool: Claude says cutscene
    static: np.ndarray            # 0..1 share of static samples
    menu: np.ndarray              # bool: Claude says menu/map/loading
    total: np.ndarray             # 0..1 combined activity
    weights: dict = field(default_factory=dict)

    def cut_indicator(self) -> np.ndarray:
        return (self.letterbox >= 0.5) | self.cut_ai


def _frame_index_per_second(frames: list[VisualFrame], n: int, max_dist: float = 2.0) -> np.ndarray:
    idx = np.full(n, -1, dtype=int)
    if not frames:
        return idx
    ft = np.array([f.t for f in frames], dtype=float)
    centres = np.arange(n) + 0.5
    nearest = np.abs(centres[:, None] - ft[None, :]).argmin(axis=1)
    dist = np.abs(centres - ft[nearest])
    idx[dist <= max_dist] = nearest[dist <= max_dist]
    return idx


def _frame_ai_value(f: VisualFrame) -> float:
    if f.kind in MENU_KINDS:
        return 0.0
    v = clip01(f.action / 10.0)
    ev = set(f.hud_events or ())
    if ev & set(BIG_HUD):
        v = max(v, 1.0)
    if "wanted_up" in ev:
        v = max(v, 0.8)
    if f.title_text and str(f.title_text).strip():
        v = max(v, 0.7)
    return v


def activity_curve(duration: float, audio: AudioFeatures, visual: VisualFeatures, cfg) -> Activity:
    n = max(1, int(np.ceil(duration - 1e-9)))
    fac = radio_factor(cfg)

    # audio: spike / transient, masked inside radio ranges
    at = np.asarray(audio.t, dtype=float)
    a_raw = np.maximum(np.asarray(audio.spike, float), np.asarray(audio.transient, float)) if at.size else np.zeros(0)
    a_mask = np.where(in_ranges(at, audio.radio_ranges), fac, 1.0) if at.size else np.zeros(0)
    audio_raw = per_second(at, a_raw, n)
    audio_m = per_second(at, a_raw * a_mask, n)

    # CV motion features
    m = visual.motion
    mt = np.asarray(m.t, dtype=float)
    if mt.size:
        burst = np.maximum.reduce([np.asarray(m.scene_change, float), np.asarray(m.flash, float),
                                   np.asarray(m.shake, float)])
        cv_raw = np.clip(0.5 * np.asarray(m.motion, float) + 0.5 * burst, 0, 1)
        cv = per_second(mt, cv_raw, n)
        lb = per_second(mt, np.asarray(m.letterbox, dtype=float), n, how="mean")
        static = per_second(mt, np.asarray(m.static, dtype=float), n, how="mean")
    else:
        cv = np.zeros(n)
        lb = np.zeros(n)
        static = np.zeros(n)

    # frame labels (every ~3 s)
    fidx = _frame_index_per_second(visual.frames, n)
    ai = np.zeros(n)
    cut_ai = np.zeros(n, dtype=bool)
    menu = np.zeros(n, dtype=bool)
    lb_frames = np.zeros(n)
    for i, j in enumerate(fidx):
        if j < 0:
            continue
        f = visual.frames[j]
        if visual.ai_ok:
            ai[i] = _frame_ai_value(f)
            cut_ai[i] = f.from_ai and f.kind == "cutscene"
        menu[i] = f.kind in MENU_KINDS
        lb_frames[i] = 1.0 if f.letterbox else 0.0
    lb = np.maximum(lb, lb_frames)

    w = dict(DEFAULT_ACTIVITY_WEIGHTS)
    w.update({k: float(v) for k, v in (sget(cfg, "activity_weights", {}) or {}).items()})
    wa, wm, wai = w["audio"], w["motion"], (w["ai"] if visual.ai_ok else 0.0)
    s = (wa + wm + wai) or 1.0
    total = (wa * audio_m + wm * cv + wai * ai) / s + w["letterbox"] * lb
    total = total * (1.0 - 0.7 * np.maximum(static, menu.astype(float)))
    total = np.clip(total, 0.0, 1.0)
    return Activity(n=n, duration=float(duration), audio=audio_m, audio_raw=audio_raw, cv=cv, ai=ai,
                    letterbox=lb, cut_ai=cut_ai, static=static, menu=menu, total=total,
                    weights={"audio": wa / s, "motion": wm / s, "ai": wai / s, "letterbox": w["letterbox"]})


# --------------------------------------------------------------------- peaks and windows
def find_peaks(curve: np.ndarray, min_gap_s: float, min_level: float) -> list[int]:
    """Greedy non-maximum suppression: strongest seconds first, at least min_gap_s apart.

    Returned strongest first.
    """
    c = np.asarray(curve, dtype=float).copy()
    peaks: list[int] = []
    gap = max(1, int(round(min_gap_s)))
    while c.size:
        i = int(np.argmax(c))
        if c[i] < min_level:
            break
        peaks.append(i)
        c[max(0, i - gap + 1): i + gap] = -np.inf
    return peaks


@dataclass
class Window:
    ws: float                     # chunk-relative start (may be < 0: previous chunk)
    we: float                     # chunk-relative end (may be > duration: next chunk)
    peak: float                   # chunk-relative peak second
    track: str = "gameplay"
    strength: float = 0.0         # activity at the peak


def gameplay_window(act: Activity, peak: int, cfg, chunk_start_s: float) -> tuple[float, float]:
    pre = float(sget(cfg, "pre_roll_s", 6))
    cmin = float(sget(cfg, "clip_min_s", 20))
    cmax = float(sget(cfg, "clip_max_s", 45))
    frac = float(sget(cfg, "sustain_frac", 0.5))
    ws = peak - pre
    if chunk_start_s + ws < 0:               # never before stream second 0
        ws = -chunk_start_s
    we = ws + cmin
    level = frac * float(act.total[peak])
    while we - ws < cmax:
        i = int(np.floor(we))
        if i >= act.n or i < 0 or act.total[i] < level:
            break
        we = min(ws + cmax, we + 1.0)
    return float(ws), float(we)


def _overlap(a: Window, b: Window) -> float:
    inter = min(a.we, b.we) - max(a.ws, b.ws)
    if inter <= 0:
        return 0.0
    return inter / max(1e-6, min(a.we - a.ws, b.we - b.ws))


def detect_windows(act: Activity, visual: VisualFeatures, cfg, chunk_start_s: float = 0.0) -> list[Window]:
    """Peaks -> windows -> track; cutscene windows trimmed; near-duplicate windows removed.

    Returned strongest peak first.
    """
    from .cutscene import cutscene_window  # local import: cutscene imports this module

    peaks = find_peaks(act.total, float(sget(cfg, "peak_min_gap_s", 20)), float(sget(cfg, "activity_min", 0.25)))
    max_overlap = float(sget(cfg, "max_window_overlap", 0.5))
    out: list[Window] = []
    for p in peaks:
        ws, we = gameplay_window(act, p, cfg, chunk_start_s)
        track = assign_track(visual, ws, we, cfg)
        if track == "cutscene":
            ws, we = cutscene_window(act, p, ws, we, cfg, chunk_start_s)
        w = Window(ws=ws, we=we, peak=float(p), track=track, strength=float(act.total[p]))
        if any(_overlap(w, o) > max_overlap for o in out):
            continue
        out.append(w)
    return out
