"""9:16 layout: blurred zoomed background + full frame centred + hook PNG, all on the GPU.

Uses exactly the graph verified in docs/CONTRACTS.md (spike findings):
  input A: h264_cuvid -crop (centre 9:16 slice, cropped by the decoder)  -> shrink + enlarge = blur
  input B: normal cuda decode                                              -> fg scaled to width
  hook PNG (single frame, no -loop) overlaid on the bg BEFORE the fg overlay.
overlay_cuda output is coded 1088 wide, fixed with -bsf:v h264_metadata=crop_right=N.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import ffmpeg

CUVID = {"h264": "h264_cuvid", "hevc": "hevc_cuvid", "vp9": "vp9_cuvid", "av1": "av1_cuvid",
         "mpeg2video": "mpeg2_cuvid", "vp8": "vp8_cuvid", "mpeg4": "mpeg4_cuvid"}


@dataclass
class SourceInfo:
    width: int
    height: int
    codec: str
    has_audio: bool


@dataclass
class Geometry:
    out_w: int
    out_h: int
    crop_l: int
    crop_r: int
    crop_w: int
    fg_w: int
    fg_h: int
    fg_y: int
    blur_w: int
    blur_h: int


def probe_source(path: str) -> SourceInfo:
    info = ffmpeg.probe(path)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if v is None:
        raise ffmpeg.FfmpegError(f"no video stream in {path}")
    has_a = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return SourceInfo(int(v["width"]), int(v["height"]), str(v.get("codec_name", "h264")), has_a)


def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def geometry(src: SourceInfo, out_w: int = 1080, out_h: int = 1920, blur=(54, 96)) -> Geometry:
    w, h = src.width, src.height
    cw = min(w, int(round(h * out_w / out_h)))       # 9:16 slice of the source, full height
    l = (w - cw) // 2
    r = w - cw - l
    fg_h = min(out_h, _even(out_w * h / w))
    fg_y = ((out_h - fg_h) // 2) // 2 * 2
    if isinstance(blur, (int, float)):               # a single number = shrink factor
        bw, bh = _even(out_w / float(blur)), _even(out_h / float(blur))
    else:
        bw, bh = _even(blur[0]), _even(blur[1])
    return Geometry(out_w, out_h, l, r, cw, out_w, fg_h, fg_y, bw, bh)


def decoder_for(codec: str) -> str:
    try:
        return CUVID[codec]
    except KeyError:
        raise ffmpeg.FfmpegError(f"no NVDEC (cuvid) decoder mapped for codec {codec!r}") from None


def video_inputs(src_args: list[str], seek: float, dur: float, codec: str, g: Geometry) -> list[str]:
    """Input 0 (bg, decoder crop) and input 1 (fg). Same source, same seek."""
    ss, t = f"{seek:.3f}", f"{dur:.3f}"
    return [*ffmpeg.GPU_DECODE, "-c:v", decoder_for(codec), "-crop", f"0x0x{g.crop_l}x{g.crop_r}",
            "-ss", ss, "-t", t, *src_args,
            *ffmpeg.GPU_DECODE, "-ss", ss, "-t", t, *src_args]


def video_filter(g: Geometry, hook_idx: int | None, hook_y: int) -> str:
    """filter_complex video chains ending in [vout]."""
    parts = [f"[0:v]scale_cuda={g.blur_w}:{g.blur_h}:format=yuv420p,"
             f"scale_cuda={g.out_w}:{g.out_h}:interp_algo=bicubic,setsar=1[bg]"]
    base = "bg"
    if hook_idx is not None:
        parts.insert(0, f"[{hook_idx}:v]format=yuva420p,hwupload_cuda[hk]")
        parts.append(f"[bg][hk]overlay_cuda=0:{hook_y}[bgh]")
        base = "bgh"
    parts.append(f"[1:v]scale_cuda={g.fg_w}:{g.fg_h}:format=yuv420p[fg]")
    parts.append(f"[{base}][fg]overlay_cuda=0:{g.fg_y}[vout]")
    return ";".join(parts)


def hook_filter(out_w: int, out_h: int, hook_y: int, hook_idx: int = 1) -> str:
    """Pass 2: overlay the hook PNG on an already-rendered 9:16 clip (input 0 = clip, input
    `hook_idx` = PNG). A SINGLE overlay_cuda on a decoded file is reliable; chaining it after the
    bg+fg overlay is not (see render_video_filter note in produce.py): the first overlay's output
    frames change size/format and the second CUDA filter rejects them at some seek points.
    The decoded clip is NV12, overlay_cuda only takes yuva420p on yuv420p, hence the convert."""
    return (f"[0:v]scale_cuda={out_w}:{out_h}:format=yuv420p[m];"
            f"[{hook_idx}:v]format=yuva420p,hwupload_cuda[hk];"
            f"[m][hk]overlay_cuda=0:{hook_y}[vout]")


def encoder_args(cfg_prod, crop_right: int = 8, cq: int | None = None) -> list[str]:
    """NVENC args. `cq` overrides the configured quality (used for the high-quality intermediate
    of a two-pass render, where the rate cap is also dropped)."""
    nv = cfg_prod.get("nvenc", {}) or {}
    args = ["-c:v", "h264_nvenc", "-preset", str(nv.get("preset", "p5")),
            "-cq", str(cq if cq is not None else nv.get("cq", 23))]
    if nv.get("maxrate") and cq is None:
        args += ["-b:v", "0", "-maxrate", str(nv["maxrate"]), "-bufsize", str(nv.get("bufsize") or nv["maxrate"])]
    elif cq is not None:
        args += ["-b:v", "0"]
    if crop_right > 0:
        args += ["-bsf:v", f"h264_metadata=crop_right={crop_right}"]
    return args
