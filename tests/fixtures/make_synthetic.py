"""Synthetic 'stream chunk' with known moments, for tests (CPU libx264 is fine for FIXTURES only).

Default 300 s timeline (scaled proportionally when duration differs):
    0-60     gameplay: moving test pattern, quiet noise floor
    75-85    gunfire: loud noise bursts (8 Hz on/off)              -> audio spike + transients
    100-160  radio: steady tonal chord                              -> music/radio range
    170-230  cutscene: letterbox bars + swelling music              -> cutscene track, swell
    240-270  loading: static black screen, silence                  -> menu/loading penalty
    285-288  explosion: one big decaying bang
Ground truth written next to the video as <name>.truth.json.

Usage:  python tests/fixtures/make_synthetic.py OUT.ts [--duration 300]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clipper.env import subprocess_env  # noqa: E402
from clipper.ffmpeg import FFMPEG  # noqa: E402

BASE = 300.0
EVENTS = {
    "gunfire": (75, 85),
    "radio": (100, 160),
    "cutscene": (170, 230),
    "loading": (240, 270),
    "explosion": (285, 288),
}


def truth(duration: float) -> dict:
    k = duration / BASE
    return {"duration": duration, "events": {n: [a * k, b * k] for n, (a, b) in EVENTS.items()}}


def make(out: Path, duration: float = 300.0, width: int = 1280, height: int = 720) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tr = truth(duration)
    ev = tr["events"]
    g0, g1 = ev["gunfire"]
    r0, r1 = ev["radio"]
    c0, c1 = ev["cutscene"]
    l0, l1 = ev["loading"]
    e0, e1 = ev["explosion"]
    bar = int(height * 0.12)
    vf = (f"drawbox=x=0:y=0:w=iw:h={bar}:color=black:t=fill:enable='between(t,{c0},{c1})',"
          f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black:t=fill:enable='between(t,{c0},{c1})',"
          f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,{l0},{l1})'")
    floor = f"0.01*(random(0)*2-1)*(1-between(t,{l0},{l1}))"
    gun = f"0.8*(random(1)*2-1)*gt(sin(2*PI*8*t),0.6)*between(t,{g0},{g1})"
    radio = f"0.15*(sin(2*PI*440*t)+sin(2*PI*554.37*t)+sin(2*PI*659.25*t))*between(t,{r0},{r1})"
    swell = f"(0.05+0.45*(t-{c0})/({c1}-{c0}))*(sin(2*PI*220*t)+0.5*sin(2*PI*330*t))*between(t,{c0},{c1})"
    boom = f"0.95*(random(2)*2-1)*exp(-(t-{e0})*2)*between(t,{e0},{e1})"
    expr = f"{floor}+{gun}+{radio}+{swell}+{boom}"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "lavfi", "-i", f"testsrc2=s={width}x{height}:r=30:d={duration}",
           "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=48000:d={duration}",
           "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-g", "60", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-f", "mpegts", str(out)]
    subprocess.run(cmd, check=True, env=subprocess_env())
    (out.with_suffix(".truth.json")).write_text(json.dumps(tr, indent=1), encoding="utf-8")
    return tr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--duration", type=float, default=300.0)
    a = ap.parse_args()
    print(json.dumps(make(Path(a.out), a.duration), indent=1))
