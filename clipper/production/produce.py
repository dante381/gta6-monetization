"""produce_clip: candidate window -> finished 1080x1920 Short (+ JSON sidecar) in one GPU ffmpeg pass.

Video: GPU only (NVDEC decode, scale_cuda, overlay_cuda, h264_nvenc), never a CPU fallback;
run_gpu waits and retries while the GPU is unavailable. Audio: CPU filters in the same command.
Output is written under DATA.work and moved into ready/ or backup/ only when complete.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable

from .. import ffmpeg, log as evlog, paths
from ..models import Candidate, Chunk, ClipSidecar
from . import audio_mix, cut, hooktext, layout, music

log = logging.getLogger(__name__)


class ProductionError(RuntimeError):
    pass


def _prod(cfg):
    return cfg.get("production", {}) or {}


def _out_dir(route: str) -> Path:
    return paths.DATA.ready if route == "publish" else paths.DATA.backup


def _probe_video(path: Path) -> dict:
    info = ffmpeg.probe(path)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    dur = float(info.get("format", {}).get("duration") or v.get("duration") or 0.0)
    return {"width": int(v.get("width", 0)), "height": int(v.get("height", 0)),
            "coded_width": int(v.get("coded_width", 0) or 0), "codec": v.get("codec_name"),
            "duration": dur, "audio_codec": a.get("codec_name"),
            "sample_rate": int(a.get("sample_rate", 0) or 0), "channels": int(a.get("channels", 0) or 0)}


def build_command(cand: Candidate, chunks: list[Chunk], cfg, work: Path) -> tuple[list[str], dict]:
    """Return (ffmpeg args without the 'ffmpeg' prefix, info dict). Writes hook PNG / concat list to `work`."""
    p = _prod(cfg)
    src = cut.plan_cut(cand.start_s, cand.end_s, chunks, work)
    info = layout.probe_source(src.probe_path)
    g = layout.geometry(info, int(p.get("width", 1080)), int(p.get("height", 1920)),
                        p.get("bg_blur_size", [54, 96]))
    dur = src.duration_s
    args = layout.video_inputs(src.input_args, src.seek_s, dur, info.codec, g)
    n_in = 2

    hook_idx = None
    hook_y = int(p.get("hook_top_px", 220))
    hook_png = hooktext.render_hook(cand.meta.hook if cand.meta else "", cfg, work / "hook.png")
    if hook_png is not None:
        from PIL import Image
        with Image.open(hook_png) as im:
            ph = im.height
        hook_y = max(0, min(hook_y, g.fg_y - 8 - ph))   # stay in the bg area above the fg
        hook_y -= hook_y % 2
        args += ["-i", str(hook_png)]                    # single frame, NO -loop (see CONTRACTS)
        hook_idx = n_in
        n_in += 1

    ranges = audio_mix.clip_ranges(cand.radio_ranges, cand.start_s, dur)
    duck_types = {str(s).lower() for s in (p.get("duck_scene_types") or [])}
    duck_tracks = {str(s).lower() for s in (p.get("duck_tracks") or ["cutscene"])}
    duck = bool(cand.scene_tag) and str(cand.scene_tag).lower() in duck_types \
        and str(cand.track).lower() in duck_tracks
    pick = music.pick_track(cand.id, dur) if (ranges or duck) else None
    music_in = None
    if pick is not None:
        if pick.loop:
            args += ["-stream_loop", "-1"]
        else:
            args += ["-ss", f"{pick.offset_s:.2f}"]
        args += ["-t", f"{dur + 1:.3f}", "-i", str(pick.path)]
        music_in = f"{n_in}:a"
        n_in += 1
    if info.has_audio:
        game_in = "1:a"
    else:                                               # silent source: synthesize silence
        args += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        game_in = f"{n_in}:a"
        n_in += 1
    ap = audio_mix.build_audio(game_in, music_in, ranges, dur, duck=duck,
                               fade_s=float(p.get("audio_fade_s", 1.0)),
                               music_volume=float(p.get("music_volume", 0.8)),
                               duck_volume=float(p.get("duck_volume", 0.25)))
    graph = layout.video_filter(g, hook_idx, hook_y) + ";" + ap.filter
    args += ["-filter_complex", graph, "-map", "[vout]", "-map", f"[{ap.label}]",
             *layout.encoder_args(p, int(p.get("crop_right_px", 8))),
             "-c:a", "aac", "-b:a", str(p.get("audio_bitrate", "192k")), "-ar", "48000", "-ac", "2",
             "-t", f"{dur:.3f}", "-map_metadata", "-1", "-movflags", "+faststart", "-f", "mp4"]
    meta = {"geometry": g, "source": info, "seek_s": src.seek_s, "duration_s": dur, "hook_png": hook_png,
            "hook_y": hook_y, "music": pick, "music_used": ap.music_used, "ducked": ap.ducked,
            "radio_ranges": ranges, "concat": src.concat_list is not None, "filter": graph}
    return args, meta


def _gpu(args: list[str], cfg, on_tick, timeout: float) -> ffmpeg.FfResult:
    p = _prod(cfg)
    return ffmpeg.run_gpu(args, on_tick=on_tick, timeout=timeout,
                          retry_wait_s=float(p.get("gpu_retry_wait_s", 30)),
                          max_wait_s=p.get("gpu_max_wait_s"))


def _replace(src: Path, dst: Path) -> None:
    try:
        os.replace(src, dst)
    except OSError:                                     # different volume: copy then rename
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        src.unlink(missing_ok=True)


def produce_clip(cand: Candidate, chunks: list[Chunk], cfg,
                 on_tick: Callable[[], None] | None = None) -> ClipSidecar:
    """Render `cand` from `chunks` to <DATA.ready|DATA.backup>/<cand.id>.mp4 + .json; return the sidecar."""
    t0 = time.monotonic()
    p = _prod(cfg)
    out_dir = _out_dir(cand.route)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = out_dir / f"{cand.id}.mp4"
    final_json = out_dir / f"{cand.id}.json"
    work = paths.DATA.work / f"{cand.id}.{os.getpid()}.{int(time.time() * 1000)}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        args, meta = build_command(cand, chunks, cfg, work)
        dur = meta["duration_s"]
        timeout = float(p.get("render_timeout_s") or max(180.0, dur * 8))
        render = work / "render.mp4"
        log.info("producing %s: %.1fs from %d chunk(s), seek %.2f, hook=%s music=%s ducked=%s",
                 cand.id, dur, len(chunks), meta["seek_s"], bool(meta["hook_png"]),
                 meta["music"].name if meta["music_used"] else None, meta["ducked"])
        _gpu([*args, str(render)], cfg, on_tick, timeout)
        if on_tick:
            on_tick()

        info = _probe_video(render)
        W, H = int(p.get("width", 1080)), int(p.get("height", 1920))
        if info["width"] != W:
            coded = info["coded_width"] or info["width"] + int(p.get("crop_right_px", 8))
            fix = coded - W
            if fix < 0:
                raise ProductionError(f"render is {info['width']} wide (coded {coded}), cannot crop to {W}")
            log.warning("render width %d (coded %d): remuxing with crop_right=%d", info["width"], coded, fix)
            fixed = work / "fixed.mp4"
            _gpu(["-i", str(render), "-map", "0", "-c", "copy", "-bsf:v", f"h264_metadata=crop_right={fix}",
                  "-movflags", "+faststart", "-f", "mp4", str(fixed)], cfg, on_tick, 120)
            render = fixed
            info = _probe_video(render)
        tol = max(1.0, 0.05 * dur)
        if (info["width"], info["height"]) != (W, H) or info["codec"] != "h264" \
                or abs(info["duration"] - dur) > tol:
            raise ProductionError(f"bad render for {cand.id}: {info} (wanted {W}x{H} h264 ~{dur:.2f}s)")

        _replace(render, final_mp4)
        pick = meta["music"]
        sc = ClipSidecar(
            clip_id=cand.id, clip_path=str(final_mp4), track=cand.track, route=cand.route,
            final_score=cand.final_score, signal_score=cand.signal_score,
            title=cand.meta.title, description=cand.meta.description, hashtags=list(cand.meta.hashtags),
            hook=cand.meta.hook, spoiler=cand.meta.spoiler, duration_s=round(info["duration"], 3),
            start_s=cand.start_s, end_s=cand.end_s,
            music_track=pick.name if (pick is not None and meta["music_used"]) else None,
            created_at=time.time(), candidate=cand.to_dict())
        tmp_json = work / "sidecar.json"
        tmp_json.write_text(json.dumps(sc.to_dict(), indent=1, default=str), encoding="utf-8")
        _replace(tmp_json, final_json)
        render_s = round(time.monotonic() - t0, 2)
        evlog.event("clip_produced", clip_id=cand.id, route=cand.route, path=str(final_mp4),
                    duration_s=sc.duration_s, render_s=render_s, music_track=sc.music_track,
                    ducked=meta["ducked"], radio_ranges=meta["radio_ranges"], hook=cand.meta.hook,
                    chunks=[c.idx for c in chunks], final_score=cand.final_score)
        log.info("clip %s ready in %.1fs -> %s", cand.id, render_s, final_mp4)
        return sc
    except BaseException:
        # never leave something that looks finished: the mp4 only counts with its sidecar
        if final_mp4.exists() and not final_json.exists():
            final_mp4.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
