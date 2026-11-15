"""Penalties (points subtracted from the signal score): menu/loading, radio, repeats."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..models import Candidate, VisualFeatures
from .candidates import MENU_KINDS, motion_slice, sget, weighted_score


def penalty_points(cfg, name: str, default: float) -> float:
    return float((sget(cfg, "penalties", {}) or {}).get(name, default))


def menu_loading_share(visual: VisualFeatures, ws: float, we: float) -> float:
    """Max of: share of window frames labelled menu/map/loading, share of CV-static motion samples."""
    shares = [0.0]
    fr = [f for f in visual.frames if ws <= f.t < we]
    if fr:
        shares.append(sum(1 for f in fr if f.kind in MENU_KINDS) / len(fr))
    ms = motion_slice(visual, ws, we)
    if ms.any():
        shares.append(float(np.asarray(visual.motion.static, dtype=bool)[ms].mean()))
    return max(shares)


def menu_loading_penalty(visual: VisualFeatures, ws: float, we: float, cfg) -> float:
    return penalty_points(cfg, "menu_loading", 30) if menu_loading_share(visual, ws, we) > 0.5 else 0.0


def radio_penalty(raw: dict, scored: dict, weights: dict, available: Iterable[str]) -> float:
    """Points lost because audio_spike / sound_type were multiplied by radio_audio_factor."""
    av = set(available)
    return max(0.0, weighted_score(raw, weights, av) - weighted_score(scored, weights, av))


def is_repeat(c: Candidate, stronger: list[Candidate], state, stream_id: str, cfg,
              exclude_ids: set[str] = frozenset()) -> str | None:
    """Reason string if `c` repeats a selected clip / a stronger candidate of this chunk, else None."""
    win = float(sget(cfg, "repeat_window_s", 60))
    scene_win = float(sget(cfg, "scene_repeat_window_s", win))
    for o in stronger:
        if abs(o.peak_s - c.peak_s) <= win:
            return f"near {o.id}"
        if c.scene_tag and o.scene_tag == c.scene_tag and abs(o.peak_s - c.peak_s) <= scene_win:
            return f"same scene as {o.id}"
    if state is None:
        return None
    for o in state.selected_near(stream_id, c.peak_s, win):
        if o.id not in exclude_ids:
            return f"near selected {o.id}"
    if c.scene_tag:
        for o in state.selected_near(stream_id, c.peak_s, scene_win):
            if o.id not in exclude_ids and o.scene_tag == c.scene_tag:
                return f"same scene as selected {o.id}"
    return None
