"""Optional AI judge: full-res frames + detected events -> claude -p (schema 'judge').

Memory rule: never a claude call and a GPU ffmpeg job at the same time. `run_judges` grabs the
frames for every gated candidate first, then calls claude one candidate at a time.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from string import Template
from typing import Callable

from .. import claude_cli, ffmpeg, paths
from .. import log as eventlog
from ..models import Candidate, Chunk
from .candidates import sget

log = logging.getLogger(__name__)

ROUTES = ("drop", "backup", "publish")


# --------------------------------------------------------------------- formulas
def final_score(signal: float, rating: float | None, judge_weight: float) -> float:
    """final = (1 - w) * signal + w * rating * 10; no rating -> final = signal."""
    if rating is None:
        return float(signal)
    w = float(judge_weight)
    return max(0.0, min(100.0, (1.0 - w) * float(signal) + w * float(rating) * 10.0))


def route_for(score: float, cfg) -> str:
    th = cfg.locked.thresholds
    if score < float(th.backup):
        return "drop"
    if score < float(th.publish):
        return "backup"
    return "publish"


def gated(c: Candidate, cfg) -> bool:
    return c.signal_score >= float(cfg.locked.thresholds.judge_gate)


# --------------------------------------------------------------------- frames
def judge_dir(cand: Candidate) -> Path:
    return paths.DATA.work / "judge" / cand.id


def judge_times(cand: Candidate, chunk: Chunk, chunk_dur: float, n: int) -> list[float]:
    """n chunk-relative times spread across the window; times outside this chunk are skipped."""
    rel0 = cand.start_s - chunk.start_s
    dur = max(0.1, cand.end_s - cand.start_s)
    out = []
    for k in range(max(1, n)):
        t = rel0 + (k + 0.5) * dur / n
        if 0.0 <= t < chunk_dur - 0.05:
            out.append(round(t, 2))
    return out


def grab_judge_frames(cand: Candidate, chunk: Chunk, chunk_dur: float, cfg,
                      on_tick: Callable[[], None] | None = None) -> list[Path]:
    times = judge_times(cand, chunk, chunk_dur, int(sget(cfg, "judge_frames", 4)))
    d = judge_dir(cand)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    if not times or not chunk.path:
        return []
    width = int(sget(cfg, "judge_frame_width", 1280))
    return [Path(p).resolve() for p in ffmpeg.grab_frames(chunk.path, times, d, width=width, prefix="judge",
                                                          on_tick=on_tick)]


# --------------------------------------------------------------------- prompt / call
def build_prompt(cand: Candidate, frame_paths: list[Path], cfg) -> str:
    game = str(cfg.publish.game_name)
    tmpl = Template(claude_cli.load_prompt("judge"))
    frame_lines = "\n".join(f"- {Path(p).resolve()}" for p in frame_paths) or "- (no frames)"
    signals = ", ".join(f"{k}={v:.2f}" for k, v in sorted(cand.signals.items()))
    return tmpl.safe_substitute(
        game_name=game,
        track=cand.track,
        duration=f"{cand.duration_s:.0f}",
        scene_tag=cand.scene_tag or "unknown",
        events="; ".join(cand.events) or "none detected",
        signals=signals or "none",
        signal_score=f"{cand.signal_score:.0f}",
        frame_list=frame_lines,
        n_frames=str(len(frame_paths)),
        base_hashtags=" ".join(cfg.publish.base_hashtags or []),
    )


def _validate(res) -> dict:
    if not isinstance(res, dict):
        raise claude_cli.ClaudeError(f"judge returned {type(res).__name__}, not an object")
    try:
        r = int(round(float(res.get("rating"))))
    except (TypeError, ValueError) as e:
        raise claude_cli.ClaudeError(f"judge rating missing/invalid: {res.get('rating')!r}") from e
    out = dict(res)
    out["rating"] = max(1, min(10, r))
    return out


def call_judge(cand: Candidate, frame_paths: list[Path], cfg, on_tick: Callable[[], None] | None = None) -> dict:
    prompt = build_prompt(cand, frame_paths, cfg)
    cl = cfg.get("claude", {}) or {}
    res = claude_cli.call(prompt, schema="judge", effort=cl.get("effort_judge", "medium"),
                          model=cl.get("model"), add_dirs=[judge_dir(cand).resolve()],
                          timeout_s=float(cl.get("timeout_s", 300)), on_tick=on_tick)
    return _validate(res)


def run_judges(cands: list[Candidate], chunk: Chunk, chunk_dur: float, cfg,
               on_tick: Callable[[], None] | None = None) -> None:
    """Fill cand.judge for gated candidates (in place). Never raises on Claude/ffmpeg trouble."""
    todo = [c for c in cands if gated(c, cfg)]
    if not todo:
        return
    # phase 1: GPU (frames), phase 2: claude, never overlapping
    frames: dict[str, list[Path]] = {}
    for c in todo:
        try:
            frames[c.id] = grab_judge_frames(c, chunk, chunk_dur, cfg, on_tick)
        except ffmpeg.GpuUnavailable:
            raise
        except (ffmpeg.FfmpegError, OSError) as e:
            log.warning("judge frames failed for %s: %s", c.id, e)
            frames[c.id] = []
        if on_tick:
            on_tick()
    unavailable = False
    for c in todo:
        fp = frames.get(c.id) or []
        if not fp:
            log.info("no judge frames for %s; final = signal", c.id)
            continue
        if unavailable:
            continue
        try:
            c.judge = call_judge(c, fp, cfg, on_tick)
            log.info("judge %s rating=%s (%s)", c.id, c.judge.get("rating"), c.judge.get("reason", ""))
        except claude_cli.ClaudeUnavailable as e:
            unavailable = True
            log.warning("claude unavailable for judge (%s); remaining candidates use signal score", e)
            eventlog.event("claude_unavailable", stage="judge", candidate=c.id, error=str(e)[:300])
        except claude_cli.ClaudeError as e:
            log.warning("judge failed for %s: %s", c.id, e)
            eventlog.event("judge_error", candidate=c.id, error=str(e)[:300])
        if on_tick:
            on_tick()
