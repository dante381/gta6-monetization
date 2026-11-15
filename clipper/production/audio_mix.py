"""Audio branch of the render filter_complex (CPU audio filters; fine per the GPU-only rule).

- No radio, no duck: game audio as is (resampled to stereo 48 kHz).
- Radio ranges (clip-relative): game gain ramps to 0 over `audio_fade_s` *outside* each range,
  so the radio is fully muted inside it; the music gain is the mirror image (crossfade).
- Ducked scene: game at `duck_volume` for the whole clip, music at `music_volume` for the
  whole clip with fade in/out at the clip edges (radio ranges still muted).
- No music: radio ranges are just muted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FMT = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"


@dataclass
class AudioPlan:
    filter: str                         # filter_complex chains ending in [aout]
    label: str = "aout"
    music_used: bool = False
    ducked: bool = False
    radio_ranges: list[tuple[float, float]] = field(default_factory=list)


def clip_ranges(radio_ranges, start_s: float, duration: float, min_len: float = 0.05) -> list[tuple[float, float]]:
    """Stream-second ranges -> sorted, merged, clip-relative ranges inside [0, duration]."""
    out: list[tuple[float, float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in (radio_ranges or [])):
        a, b = max(0.0, a - start_s), min(duration, b - start_s)
        if b - a < min_len:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _f(x: float) -> str:
    s = f"{float(x):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def radio_weight_expr(ranges: list[tuple[float, float]], fade: float) -> str:
    """ffmpeg expression in t: 1 inside any range, ramping 0->1 over `fade` s before/after it."""
    if not ranges:
        return "0"
    fade = max(0.01, float(fade))
    terms = [f"clip((t-({_f(a - fade)}))/{_f(fade)},0,1)*clip(({_f(b + fade)}-t)/{_f(fade)},0,1)"
             for a, b in ranges]
    return f"clip({'+'.join(terms)},0,1)"


def build_audio(game_in: str, music_in: str | None, ranges: list[tuple[float, float]], duration: float,
                *, duck: bool, fade_s: float, music_volume: float, duck_volume: float) -> AudioPlan:
    """Return the audio filter chains. `game_in`/`music_in` are pad labels like '1:a'."""
    w = radio_weight_expr(ranges, fade_s)
    use_music = music_in is not None and (bool(ranges) or duck)
    game_gain = "1"
    if ranges:
        game_gain = f"1-{w}"
    if duck and use_music:
        game_gain = f"{duck_volume:g}*({game_gain})"
    chains = []
    g = f"[{game_in}]asetpts=PTS-STARTPTS,{FMT}"
    if game_gain != "1":
        g += f",volume='{game_gain}':eval=frame"
    if not use_music:
        chains.append(g + "[aout]")
        return AudioPlan(";".join(chains), music_used=False, ducked=False, radio_ranges=ranges)
    chains.append(g + "[ag]")
    m = f"[{music_in}]atrim=0:{_f(duration)},asetpts=PTS-STARTPTS,{FMT}"
    if duck:
        f = max(0.01, min(float(fade_s), duration / 4))
        m += (f",volume={music_volume:g},afade=t=in:st=0:d={_f(f)}"
              f",afade=t=out:st={_f(max(0.0, duration - f))}:d={_f(f)}")
    else:
        m += f",volume='{music_volume:g}*{w}':eval=frame"
    chains.append(m + "[am]")
    chains.append("[ag][am]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]")
    return AudioPlan(";".join(chains), music_used=True, ducked=duck, radio_ranges=ranges)
