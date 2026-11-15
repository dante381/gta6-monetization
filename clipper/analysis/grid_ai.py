"""Claude grid labelling: ONE claude -p call per chunk covering all its grid images.

The prompt (prompts/grid.md) lists the absolute grid paths (Claude opens them with the Read tool),
a `TIMESTAMPS: 0,3,6,...` line with every tile timestamp (contract), the label vocabulary and,
optionally, HUD reference images from paths.HUD_REFS.

label_grids() returns only the frames Claude labelled (from_ai=True, letterbox unset); vision.py
merges them with the CV measures and fills any missing timestamps.
Raises ClaudeUnavailable / ClaudeError; it never falls back by itself.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import claude_cli, paths
from ..claude_cli import ClaudeError, ClaudeUnavailable
from ..models import HUD_EVENTS, VisualFrame

logger = logging.getLogger(__name__)

FRAME_KINDS = ("gameplay", "cutscene", "menu", "loading", "map", "other")
IMAGE_EXT = (".png", ".jpg", ".jpeg")


def _fmt_t(t: float) -> str:
    return str(int(round(t))) if abs(t - round(t)) < 1e-6 else f"{t:.1f}"


def hud_ref_images(cfg) -> list[Path]:
    d = Path(paths.HUD_REFS)
    if not d.is_dir():
        return []
    n = int(cfg.analysis.get("hud_ref_max_images", 6))
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)[:max(0, n)]


def build_prompt(grids: Sequence[tuple[str, Sequence[float]]], cfg, hud_refs: bool = True
                 ) -> tuple[str, list[Path]]:
    """Fill prompts/grid.md. Returns (prompt, add_dirs)."""
    a = cfg.analysis
    cols, rows = int(a.grid_cols), int(a.grid_rows)
    all_ts = [t for _, ts in grids for t in ts]
    lines = []
    for i, (g, ts) in enumerate(grids, 1):
        gp = Path(g).resolve()
        lines.append(f"{i}. {gp}  (tiles t={_fmt_t(ts[0])} to t={_fmt_t(ts[-1])}, {len(ts)} tiles)")
    refs = hud_ref_images(cfg) if hud_refs else []
    if refs:
        ref_txt = ("Reference examples of HUD events (screenshots from this game; the file name says what "
                   "each one shows). Open them with the Read tool first and use them to recognise the same "
                   "HUD elements in the tiles:\n" + "\n".join(f"- {p.resolve()}" for p in refs))
    else:
        ref_txt = "(No HUD reference images are available; rely on the descriptions above.)"
    try:
        game = str(cfg.publish.game_name)
    except (AttributeError, KeyError):
        game = "GTA 6"
    repl = {
        "{{GAME}}": game,
        "{{N_GRIDS}}": str(len(grids)),
        "{{N_FRAMES}}": str(len(all_ts)),
        "{{COLS}}": str(cols),
        "{{ROWS}}": str(rows),
        "{{PER_GRID}}": str(cols * rows),
        "{{INTERVAL}}": _fmt_t(float(a.frame_interval_s)),
        "{{GRID_LIST}}": "\n".join(lines),
        "{{TIMESTAMPS}}": ",".join(_fmt_t(t) for t in all_ts),
        "{{HUD_REFS}}": ref_txt,
    }
    prompt = claude_cli.load_prompt("grid")
    for k, v in repl.items():
        prompt = prompt.replace(k, v)
    dirs: list[Path] = []
    for g, _ in grids:
        d = Path(g).resolve().parent
        if d not in dirs:
            dirs.append(d)
    if refs:
        dirs.append(Path(paths.HUD_REFS).resolve())
    return prompt, dirs


def _clean_text(v: Any, max_len: int) -> str | None:
    if v is None:
        return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    if not s or s.lower() in ("null", "none", "n/a", "-"):
        return None
    return s[:max_len]


def _int(v: Any, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return lo


def parse_frames(data: Any, timestamps: Sequence[float], interval: float) -> list[VisualFrame]:
    """Validate Claude's answer and snap each entry to a known tile timestamp."""
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        raise ClaudeError(f"grid answer has no 'frames' list: {str(data)[:200]}")
    known = sorted(timestamps)
    tol = max(0.5, interval / 2.0)
    out: dict[float, VisualFrame] = {}
    for item in data["frames"]:
        if not isinstance(item, dict):
            continue
        try:
            t = float(item.get("t"))
        except (TypeError, ValueError):
            continue
        if not known:
            break
        near = min(known, key=lambda k: abs(k - t))
        if abs(near - t) > tol or near in out:
            continue
        kind = str(item.get("kind") or "other").strip().lower()
        if kind not in FRAME_KINDS:
            kind = "other"
        hud = item.get("hud_events") or []
        if isinstance(hud, str):
            hud = [hud]
        events: list[str] = []
        for e in hud:
            e = str(e).strip().lower()
            if e in HUD_EVENTS and e not in events:
                events.append(e)
        tag = _clean_text(item.get("scene_tag"), 60)
        out[near] = VisualFrame(
            t=near, kind=kind, hud_events=events,
            wanted_stars=_int(item.get("wanted_stars"), 0, 5),
            title_text=_clean_text(item.get("title_text"), 200),
            action=_int(item.get("action"), 0, 10),
            scene_tag=tag.lower() if tag else None,
            letterbox=False, from_ai=True)
    if not out:
        raise ClaudeError("grid answer matched none of the tile timestamps")
    return [out[t] for t in sorted(out)]


def label_grids(grids: Sequence[tuple[str, Sequence[float]]], cfg, on_tick: Callable[[], None] | None = None,
                hud_refs: bool = True) -> list[VisualFrame]:
    """ONE Claude call for all of a chunk's grids. Raises ClaudeUnavailable / ClaudeError."""
    if not grids:
        return []
    prompt, dirs = build_prompt(grids, cfg, hud_refs)
    c = cfg.claude
    t0 = time.monotonic()
    try:
        data = claude_cli.call(prompt, schema="grid", effort=c.effort_grid, add_dirs=dirs, model=c.model,
                               on_tick=on_tick, timeout_s=float(c.timeout_s))
    except ClaudeError:
        raise
    except OSError as e:  # claude binary missing / not startable
        raise ClaudeUnavailable(f"claude could not be started: {e}") from e
    timestamps = [t for _, ts in grids for t in ts]
    frames = parse_frames(data, timestamps, float(cfg.analysis.frame_interval_s))
    logger.info("grid call: %d/%d frames labelled in %.1fs", len(frames), len(timestamps),
                time.monotonic() - t0)
    return frames
