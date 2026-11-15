"""Highlight scoring: activity curve -> candidates -> track -> signals -> penalties -> judge -> route.

    score_chunk(chunk, audio, visual, cfg, state, on_tick=None) -> list[Candidate]

signal_score = 100 * sum(w_i * s_i) / sum(w_i)   (over the signals measurable for this chunk)
               - penalties (menu_loading, radio, repeat), clamped to 0..100
final_score  = (1 - judge_weight) * signal + judge_weight * rating * 10   (judge only if
               signal >= locked judge_gate and Claude answered; otherwise final = signal)
route        = drop < backup <= backup < publish <= publish   (cfg.locked.thresholds)
"""
from __future__ import annotations

import logging
from typing import Callable

from .. import log as eventlog
from ..models import AudioFeatures, Candidate, Chunk, VisualFeatures
from .candidates import (Activity, Window, activity_curve, assign_track, chunk_duration, detect_windows,
                         find_peaks, radio_ranges_in, sget, weighted_score)
from .cutscene import CUTSCENE_SIGNALS, cutscene_signals, cutscene_window
from .gameplay import GAMEPLAY_SIGNALS, SignalResult, gameplay_signals
from .judge import build_prompt, final_score, route_for, run_judges
from .metadata import build_meta, make_hashtags, make_hook, make_title
from .penalties import is_repeat, menu_loading_penalty, penalty_points, radio_penalty

log = logging.getLogger(__name__)

__all__ = ["score_chunk", "score_window", "activity_curve", "detect_windows", "find_peaks", "assign_track",
           "gameplay_signals", "cutscene_signals", "cutscene_window", "weighted_score", "final_score",
           "route_for", "build_meta", "build_prompt", "run_judges", "make_title", "make_hashtags", "make_hook",
           "Activity", "Window", "SignalResult"]


def _weights(cfg, track: str) -> dict:
    key = "cutscene_weights" if track == "cutscene" else "gameplay_weights"
    names = CUTSCENE_SIGNALS if track == "cutscene" else GAMEPLAY_SIGNALS
    w = dict(sget(cfg, key, {}) or {})
    return {k: float(w.get(k, 0.0)) for k in names}


def score_window(win: Window, chunk: Chunk, stream_id: str, audio: AudioFeatures, visual: VisualFeatures,
                 act: Activity, cfg, state) -> Candidate:
    """Signals + menu/radio penalties for one window (repeat penalty is applied by score_chunk)."""
    ws, we, peak = win.ws, win.we, win.peak
    if win.track == "cutscene":
        sr = cutscene_signals(ws, we, peak, audio, visual, act, cfg, state)
    else:
        sr = gameplay_signals(ws, we, peak, audio, visual, act, cfg)
    weights = _weights(cfg, win.track)
    base = weighted_score(sr.raw, weights, sr.available)
    penalties: dict[str, float] = {}
    rp = radio_penalty(sr.raw, sr.scored, weights, sr.available)
    if rp > 0.05:
        penalties["radio"] = round(rp, 2)
    mp = menu_loading_penalty(visual, ws, we, cfg)
    events = list(sr.events)
    if mp:
        penalties["menu_loading"] = mp
        events.append("menu / loading screen")
    signal = max(0.0, min(100.0, base - sum(penalties.values())))
    peak_s = chunk.start_s + peak
    return Candidate(
        id=f"{stream_id}-{chunk.idx}-{int(peak_s * 10)}",
        chunk_idx=chunk.idx,
        start_s=round(max(0.0, chunk.start_s + ws), 3),
        peak_s=round(peak_s, 3),
        end_s=round(chunk.start_s + we, 3),
        track=win.track,
        signals={k: round(float(v), 4) for k, v in sr.raw.items()},
        penalties=penalties,
        signal_score=round(signal, 2),
        events=events,
        scene_tag=sr.scene_tag,
        radio_ranges=[(round(chunk.start_s + a, 3), round(chunk.start_s + b, 3))
                      for a, b in radio_ranges_in(audio, ws, we)],
    )


def apply_repeats(cands: list[Candidate], state, stream_id: str, cfg) -> None:
    """Repeat penalty, strongest first: near an already-selected clip (state), or near / same scene
    as a stronger candidate of this chunk that is likely to be kept (signal >= backup)."""
    pts = penalty_points(cfg, "repeat", 25)
    keep_level = float(cfg.locked.thresholds.backup)
    own = {c.id for c in cands}
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: -c.signal_score):
        why = is_repeat(c, kept, state, stream_id, cfg, exclude_ids=own)
        if why:
            c.penalties["repeat"] = pts
            c.events.append(f"repeat ({why})")
            c.signal_score = round(max(0.0, c.signal_score - pts), 2)
        if c.signal_score >= keep_level:
            kept.append(c)


def score_chunk(chunk: Chunk, audio: AudioFeatures, visual: VisualFeatures, cfg, state,
                on_tick: Callable[[], None] | None = None) -> list[Candidate]:
    stream_id = chunk.stream_id or (state.current_stream() if state is not None else "") or "stream"
    dur = chunk_duration(chunk, audio, visual)
    act = activity_curve(dur, audio, visual, cfg)
    windows = detect_windows(act, visual, cfg, chunk.start_s)
    cands = [score_window(w, chunk, stream_id, audio, visual, act, cfg, state) for w in windows]
    if on_tick:
        on_tick()
    apply_repeats(cands, state, stream_id, cfg)

    run_judges(cands, chunk, dur, cfg, on_tick)

    jw = float(sget(cfg, "judge_weight", 0.3))
    for c in cands:
        rating = c.judge.get("rating") if isinstance(c.judge, dict) else None
        c.final_score = round(final_score(c.signal_score, rating, jw), 2)
        c.route = route_for(c.final_score, cfg)
        c.meta = build_meta(c, cfg)

    cands.sort(key=lambda c: c.peak_s)
    for c in cands:
        if state is not None:
            state.save_candidate(c, stream_id)
            if c.route != "drop" and c.scene_tag:
                state.mark_scene(c.scene_tag, c.peak_s)
        eventlog.event("candidate", id=c.id, stream_id=stream_id, chunk_idx=c.chunk_idx, track=c.track,
                       start_s=c.start_s, peak_s=c.peak_s, end_s=c.end_s, signal_score=c.signal_score,
                       final_score=c.final_score, route=c.route, signals=c.signals, penalties=c.penalties,
                       events=c.events, scene_tag=c.scene_tag,
                       rating=(c.judge or {}).get("rating") if isinstance(c.judge, dict) else None,
                       title=c.meta.title)
    log.info("chunk %s: %d candidates (%s)", chunk.idx, len(cands),
             ", ".join(f"{c.route}:{c.final_score:.0f}" for c in cands))
    return cands
