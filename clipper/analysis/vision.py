"""Vision analysis for one chunk: GPU decode -> small frames + motion series -> grids -> Claude labels.

Pipeline (one chunk):
  1. ONE GPU decode with two outputs (spike-verified graph):
       * every `frame_interval_s` a `frame_size` RGB frame (saved as JPEG, tiled into grids for Claude)
       * `motion_fps` tiny gray frames (`motion_size`) for the CV motion measures
     Raw frames go to temp files in DATA.frames/<stream_id>/<idx>/ and are read back one frame at a
     time, so the chunk is never held in RAM.
  2. Motion measures from the gray stream (all 0..1): motion, scene_change, flash, shake, plus the
     letterbox and static flags.
  3. Grids: cols x rows tiles, a `t=<s>` label strip above every tile (chunk-relative seconds).
  4. ONE Claude grid call for the whole chunk (grid_ai.label_grids). Claude labels are merged with
     the CV letterbox per timestamp; timestamps Claude skipped are filled from CV. If Claude is
     unavailable or answers garbage, every frame comes from CV (from_ai=False, ai_ok=False).

GPU only: decode + scaling happen on CUDA; only the finished small frames reach system memory.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import ffmpeg, log, paths
from ..claude_cli import ClaudeError
from ..models import Chunk, MotionSeries, VisualFeatures, VisualFrame

logger = logging.getLogger(__name__)

Tick = Callable[[], None] | None

# Defaults for the optional `analysis.vision_*` settings (settings.yaml may override any of them).
DEFAULTS = {
    "vision_motion_gain": 6.0,            # motion = clip(mean_abs_diff/255 * gain)
    "vision_scene_hist_bins": 32,
    "vision_scene_full": 0.5,             # histogram distance (0..1) that counts as a certain cut
    "vision_flash_gain": 5.0,             # flash = clip(brightness jump/255 * gain)
    "vision_flash_min_prev_luma": 12,     # no flash when coming out of a black frame (fade-in)
    "vision_shake_full_px": 3.0,          # global shift (px on the motion frame) that counts as 1.0
    "vision_shake_min_response": 0.15,    # phaseCorrelate confidence below this => no shift
    "vision_static_window_s": 2.0,        # near-zero motion for this long => static
    "vision_static_diff_max": 0.01,       # mean abs diff (0..1 scale) counted as "no motion"
    "vision_letterbox_mid_margin": 10,    # middle (90th pct luma) must exceed luma_max by this
    "vision_letterbox_band_pct_max": 32,  # 98th pct luma inside a bar must stay below this
    "vision_label_px": 26,                # height of the t=<s> strip above each tile
    "vision_jpeg_quality": 85,
    "vision_decode_timeout_s": 900,
}


def _opt(cfg, key: str):
    a = cfg.get("analysis", {}) if hasattr(cfg, "get") else {}
    v = a.get(key) if hasattr(a, "get") else None
    return DEFAULTS[key] if v is None else v


class _Ticker:
    """Calls on_tick at most every `every_s` seconds (heartbeat while crunching frames)."""

    def __init__(self, on_tick: Tick, every_s: float = 2.0):
        self.on_tick = on_tick
        self.every_s = every_s
        self.last = time.monotonic()

    def __call__(self, force: bool = False) -> None:
        if self.on_tick and (force or time.monotonic() - self.last >= self.every_s):
            self.last = time.monotonic()
            self.on_tick()


# --------------------------------------------------------------------------- paths
def _stream_dir(chunk: Chunk) -> str:
    return chunk.stream_id or "nostream"


def frames_dir(chunk: Chunk) -> Path:
    return paths.DATA.frames / _stream_dir(chunk) / str(chunk.idx)


def grids_dir(chunk: Chunk) -> Path:
    return paths.DATA.grids / _stream_dir(chunk) / str(chunk.idx)


def _fmt_t(t: float) -> str:
    return str(int(round(t))) if abs(t - round(t)) < 1e-6 else f"{t:.1f}"


def _clean_dir(d: Path, patterns: Sequence[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        for p in d.glob(pat):
            try:
                p.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- CV measures
def letterbox_flag(gray: np.ndarray, cfg) -> bool:
    """True when the top and bottom bands are black bars while the middle has picture.

    A fully black frame (fade, loading) is NOT letterbox.
    """
    a = cfg.analysis
    h = gray.shape[0]
    b = max(1, int(round(h * float(a.letterbox_rows_frac))))
    if h < 2 * b + 2:
        return False
    luma_max = float(a.letterbox_luma_max)
    band_pct_max = float(_opt(cfg, "vision_letterbox_band_pct_max"))
    for band in (gray[:b], gray[h - b:]):
        if band.mean() >= luma_max or np.percentile(band, 98) >= band_pct_max:
            return False
    mid = gray[b:h - b]
    return float(np.percentile(mid, 90)) > luma_max + float(_opt(cfg, "vision_letterbox_mid_margin"))


def _hist(gray: np.ndarray, bins: int) -> np.ndarray:
    h = np.bincount((gray.astype(np.uint16) * bins // 256).ravel(), minlength=bins).astype(np.float32)
    return h / max(1.0, h.sum())


class _MotionAccumulator:
    """Consumes gray motion frames one at a time; keeps only the previous frame."""

    def __init__(self, cfg, size: tuple[int, int]):
        self.cfg = cfg
        self.bins = int(_opt(cfg, "vision_scene_hist_bins"))
        self.window = cv2.createHanningWindow(size, cv2.CV_32F)
        self.prev: np.ndarray | None = None
        self.prev_hist: np.ndarray | None = None
        self.diff: list[float] = []      # raw mean abs diff, 0..1
        self.hist_d: list[float] = []    # histogram distance, 0..1
        self.jump: list[float] = []      # brightness jump, 0..1 (positive only, gated)
        self.shift: list[float] = []     # global shift magnitude (px)
        self.letterbox: list[bool] = []

    def add(self, gray: np.ndarray) -> None:
        g = gray.astype(np.float32)
        hist = _hist(gray, self.bins)
        self.letterbox.append(letterbox_flag(gray, self.cfg))
        if self.prev is None:
            self.diff.append(math.nan)
            self.hist_d.append(0.0)
            self.jump.append(0.0)
            self.shift.append(0.0)
        else:
            p = self.prev
            self.diff.append(float(np.abs(g - p).mean()) / 255.0)
            hd = 0.5 * float(np.abs(hist - self.prev_hist).sum())
            self.hist_d.append(hd)
            pm, gm = float(p.mean()), float(g.mean())
            j = (gm - pm) / 255.0 if pm >= float(_opt(self.cfg, "vision_flash_min_prev_luma")) else 0.0
            self.jump.append(max(0.0, j))
            sh = 0.0
            if p.std() > 2.0 and g.std() > 2.0 and hd < 0.5:   # no shift through hard cuts/black
                # phaseCorrelate applies the window IN PLACE: give it copies
                (dx, dy), resp = cv2.phaseCorrelate(p.copy(), g.copy(), self.window)
                if resp >= float(_opt(self.cfg, "vision_shake_min_response")):
                    sh = math.hypot(dx, dy)
            self.shift.append(sh)
        self.prev = g
        self.prev_hist = hist

    def series(self, fps: float) -> MotionSeries:
        n = len(self.diff)
        t = np.arange(n, dtype=np.float64) / fps
        diff = np.array(self.diff, dtype=np.float64)
        if n > 1:
            diff[0] = diff[1]
        elif n == 1:
            diff[0] = 0.0
        motion = np.clip(diff * float(_opt(self.cfg, "vision_motion_gain")), 0, 1)
        scene = np.clip(np.array(self.hist_d) / float(_opt(self.cfg, "vision_scene_full")), 0, 1)
        flash = np.clip(np.array(self.jump) * float(_opt(self.cfg, "vision_flash_gain")), 0, 1)
        shake = np.clip(np.array(self.shift) / float(_opt(self.cfg, "vision_shake_full_px")), 0, 1)
        # static: near-zero diff over a short centred window
        w = max(1, int(round(float(_opt(self.cfg, "vision_static_window_s")) * fps)))
        quiet = diff <= float(_opt(self.cfg, "vision_static_diff_max"))
        static = np.zeros(n, dtype=bool)
        for i in range(n):
            lo, hi = max(0, i - w // 2), min(n, i - w // 2 + w)
            static[i] = bool(quiet[lo:hi].all())   # windows are truncated at the series edges
        return MotionSeries(t=t, motion=motion.astype(np.float32), scene_change=scene.astype(np.float32),
                            flash=flash.astype(np.float32), shake=shake.astype(np.float32),
                            letterbox=np.array(self.letterbox, dtype=bool), static=static)


def _empty_motion() -> MotionSeries:
    z = np.zeros(0, dtype=np.float32)
    return MotionSeries(t=np.zeros(0), motion=z, scene_change=z, flash=z, shake=z,
                        letterbox=np.zeros(0, dtype=bool), static=np.zeros(0, dtype=bool))


# --------------------------------------------------------------------------- decode
def _decode_args(src: str, cfg, rgb_out: Path, gray_out: Path) -> list[str]:
    a = cfg.analysis
    fw, fh = (int(x) for x in a.frame_size)
    mw, mh = (int(x) for x in a.motion_size)
    interval = float(a.frame_interval_s)
    # round=up: output frame k is the input frame at (or just before) k/fps. The default (near)
    # was measured to lag ~1.3 s at fps=1/3 (frame "t=33" showed source 34.3 s).
    graph = (f"[0:v]setpts=PTS-STARTPTS,split=2[a][b];"
             f"[a]fps=1/{interval:g}:round=up,scale_cuda={fw}:{fh},hwdownload,format=nv12[fa];"
             f"[b]fps={float(a.motion_fps):g}:round=up,scale_cuda={mw}:{mh},hwdownload,format=nv12[fb]")
    return [*ffmpeg.GPU_DECODE, "-i", src, "-filter_complex", graph,
            "-map", "[fa]", "-f", "rawvideo", "-pix_fmt", "rgb24", str(rgb_out),
            "-map", "[fb]", "-f", "rawvideo", "-pix_fmt", "gray", str(gray_out)]


def extract_frames(chunk: Chunk, cfg, out_dir: str | Path, on_tick: Tick = None
                   ) -> tuple[dict[float, str], MotionSeries]:
    """GPU-decode the chunk once; save small JPEG frames to out_dir and measure motion.

    Returns ({t: jpg path}, MotionSeries). t is chunk-relative seconds (frame i at i*interval).
    """
    out_dir = Path(out_dir)
    _clean_dir(out_dir, ("*.jpg", "_*.rgb", "_*.gray"))
    a = cfg.analysis
    fw, fh = (int(x) for x in a.frame_size)
    mw, mh = (int(x) for x in a.motion_size)
    interval = float(a.frame_interval_s)
    mfps = float(a.motion_fps)
    rgb_p, gray_p = out_dir / "_frames.rgb", out_dir / "_motion.gray"
    tick = _Ticker(on_tick)
    t0 = time.monotonic()
    ffmpeg.run_gpu(_decode_args(str(chunk.path), cfg, rgb_p, gray_p), on_tick=on_tick,
                   timeout=float(_opt(cfg, "vision_decode_timeout_s")))
    dur = float(chunk.duration_s or 0.0)
    quality = int(_opt(cfg, "vision_jpeg_quality"))
    frame_paths: dict[float, str] = {}
    try:
        fsz = fw * fh * 3
        with open(rgb_p, "rb") as f:
            i = 0
            while True:
                buf = f.read(fsz)
                if len(buf) < fsz:
                    break
                t = round(i * interval, 3)
                i += 1
                if dur > 0 and t >= dur:
                    break
                p = out_dir / f"t{t:07.1f}.jpg"
                Image.frombytes("RGB", (fw, fh), buf).save(p, quality=quality)
                frame_paths[t] = str(p)
                tick()
        acc = _MotionAccumulator(cfg, (mw, mh))
        msz = mw * mh
        with open(gray_p, "rb") as f:
            j = 0
            while True:
                buf = f.read(msz)
                if len(buf) < msz:
                    break
                if dur > 0 and j / mfps >= dur:
                    break
                acc.add(np.frombuffer(buf, dtype=np.uint8).reshape(mh, mw))
                j += 1
                tick()
        motion = acc.series(mfps) if acc.diff else _empty_motion()
    finally:
        for p in (rgb_p, gray_p):
            try:
                p.unlink()
            except OSError:
                pass
    logger.info("chunk %s: %d frames, %d motion samples in %.1fs", chunk.idx, len(frame_paths),
                len(motion.t), time.monotonic() - t0)
    return frame_paths, motion


# --------------------------------------------------------------------------- CV-only frames
def _window(motion: MotionSeries, t: float, half: float) -> slice:
    i0 = int(np.searchsorted(motion.t, t - half))
    i1 = int(np.searchsorted(motion.t, t + half))
    return slice(i0, max(i0, i1))


def cv_frames(frame_paths: dict[float, str], motion: MotionSeries, cfg) -> list[VisualFrame]:
    """CV-only VisualFrame per sampled frame: letterbox from the small frame (motion series as backup),
    kind from letterbox (cutscene) / static or fully black (loading) / else gameplay, action from motion."""
    half = float(cfg.analysis.frame_interval_s) / 2.0
    out: list[VisualFrame] = []
    for t in sorted(frame_paths):
        sl = _window(motion, t, half)
        gray = cv2.imread(frame_paths[t], cv2.IMREAD_GRAYSCALE)
        black = False
        if gray is not None:
            lb = letterbox_flag(gray, cfg)
            black = float(np.percentile(gray, 98)) < float(cfg.analysis.letterbox_luma_max)
        else:
            lb = bool(motion.letterbox[sl].mean() > 0.5) if sl.stop > sl.start else False
        if sl.stop > sl.start:
            static = bool(motion.static[sl].mean() > 0.5)
            m = float(motion.motion[sl].mean())
            act = m + 0.3 * float(motion.shake[sl].mean()) + 0.3 * float(motion.scene_change[sl].max()) \
                + 0.3 * float(motion.flash[sl].max())
        else:
            static, act = False, 0.0
        static = static or black            # a fully black frame is a loading/fade frame
        kind = "cutscene" if lb else ("loading" if static else "gameplay")
        action = 0 if static else int(round(10 * min(1.0, act)))
        out.append(VisualFrame(t=t, kind=kind, action=action, letterbox=lb, from_ai=False))
    return out


# --------------------------------------------------------------------------- grids
def _font(cfg, size: int):
    cands = []
    try:
        cands.append(str(cfg.production.font_path))
    except (AttributeError, KeyError):
        pass
    cands += ["arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"]
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def build_grids(frame_paths: dict[float, str], cfg, out_dir: str | Path) -> list[tuple[str, list[float]]]:
    """Tile the frames into grids (cols x rows, row-major) with a `t=<s>` label above every tile.

    Returns [(grid jpg path, [timestamps in tile order])].
    """
    out_dir = Path(out_dir)
    _clean_dir(out_dir, ("grid_*.jpg",))
    a = cfg.analysis
    cols, rows = int(a.grid_cols), int(a.grid_rows)
    fw, fh = (int(x) for x in a.frame_size)
    lab = int(_opt(cfg, "vision_label_px"))
    font = _font(cfg, max(10, lab - 6))
    per = cols * rows
    ts = sorted(frame_paths)
    grids: list[tuple[str, list[float]]] = []
    for k in range(0, len(ts), per):
        group = ts[k:k + per]
        canvas = Image.new("RGB", (cols * fw, rows * (fh + lab)), (40, 40, 40))
        draw = ImageDraw.Draw(canvas)
        for n, t in enumerate(group):
            x, y = (n % cols) * fw, (n // cols) * (fh + lab)
            draw.rectangle([x, y, x + fw - 1, y + lab - 1], fill=(0, 0, 0))
            draw.text((x + 6, y + lab // 2), f"t={_fmt_t(t)}", fill=(255, 255, 0), font=font, anchor="lm")
            with Image.open(frame_paths[t]) as im:
                tile = im.convert("RGB")
                if tile.size != (fw, fh):
                    tile = tile.resize((fw, fh))
                canvas.paste(tile, (x, y + lab))
            # thin separator so tiles don't blend into each other
            draw.rectangle([x, y, x + fw - 1, y + fh + lab - 1], outline=(90, 90, 90))
        p = out_dir / f"grid_{len(grids):02d}.jpg"
        canvas.save(p, quality=int(_opt(cfg, "vision_jpeg_quality")))
        grids.append((str(p), list(group)))
    return grids


# --------------------------------------------------------------------------- merge + entry point
def merge_frames(cv: list[VisualFrame], ai: list[VisualFrame]) -> list[VisualFrame]:
    """Claude labels per timestamp + CV letterbox; timestamps Claude skipped stay CV-only."""
    by_t = {round(f.t, 3): f for f in ai}
    out = []
    for c in cv:
        f = by_t.get(round(c.t, 3))
        if f is None:
            out.append(c)
        else:
            f.letterbox = c.letterbox
            f.from_ai = True
            out.append(f)
    return out


def analyze_video(chunk: Chunk, cfg, on_tick: Tick = None, use_ai: bool = True,
                  hud_refs: bool = True) -> VisualFeatures:
    """Frames, motion, grids and (one) Claude grid call for a chunk. Never fails on Claude errors."""
    fdir, gdir = frames_dir(chunk), grids_dir(chunk)
    frame_paths, motion = extract_frames(chunk, cfg, fdir, on_tick)
    cvf = cv_frames(frame_paths, motion, cfg)
    grids = build_grids(frame_paths, cfg, gdir)
    if on_tick:
        on_tick()
    frames, ai_ok = cvf, False
    if use_ai and grids:
        from . import grid_ai
        try:
            ai = grid_ai.label_grids(grids, cfg, on_tick=on_tick, hud_refs=hud_refs)
            frames, ai_ok = merge_frames(cvf, ai), True
        except ClaudeError as e:
            _log_unavailable(chunk, e)
        except Exception as e:  # noqa: BLE001 - a Claude problem must never stop the pipeline
            logger.exception("grid labelling failed unexpectedly")
            _log_unavailable(chunk, e)
    return VisualFeatures(chunk_idx=chunk.idx, frames=frames, motion=motion, frame_paths=frame_paths,
                          grid_paths=[g for g, _ in grids], ai_ok=ai_ok)


def _log_unavailable(chunk: Chunk, e: BaseException) -> None:
    logger.warning("claude grid call failed for chunk %s, using CV only: %s", chunk.idx, str(e)[:300])
    log.event("claude_unavailable", stage="grid", chunk_idx=chunk.idx, stream_id=chunk.stream_id,
              error_type=type(e).__name__, error=str(e)[:500])
