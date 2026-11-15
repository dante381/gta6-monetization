"""Generate original royalty-free music beds into music/ (no licensing, nothing to claim).

These replace in-game radio in clips (see Clip production > Audio in the plan). They are plain
synthesized ambient beds: slow chord pads, a soft arpeggio and a gentle pulse, with a simple
echo. Replace or add tracks from the YouTube Audio Library (no attribution required) any time:
any .mp3/.wav/.m4a/.ogg/.flac in music/ is used.

    .venv\\Scripts\\python.exe scripts\\make_music.py [--seconds 150]
Output goes to music/ inside the project (m4a via the project's ffmpeg).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clipper import ffmpeg  # noqa: E402
from clipper.env import subprocess_env  # noqa: E402

SR = 44100


def midi_hz(n: float) -> float:
    return 440.0 * 2 ** ((n - 69) / 12)


def adsr(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    env = np.ones(n) * s
    a_n, d_n, r_n = min(a_n, n), min(d_n, max(0, n - a_n)), min(r_n, n)
    env[:a_n] = np.linspace(0, 1, a_n, endpoint=False) if a_n else env[:a_n]
    env[a_n:a_n + d_n] = np.linspace(1, s, d_n, endpoint=False) if d_n else env[a_n:a_n + d_n]
    if r_n:
        env[-r_n:] *= np.linspace(1, 0, r_n)
    return env


def tone(freq: float, dur: float, kind: str, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    if kind == "pad":
        x = sum(np.sin(2 * np.pi * freq * (1 + d) * t + rng.uniform(0, 6.28)) * a
                for d, a in ((-0.004, 0.5), (0.0, 1.0), (0.005, 0.5)))
        x += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
        return x * adsr(n, 1.2, 0.8, 0.8, 1.5) * 0.18
    if kind == "pluck":
        x = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        return x * np.exp(-t * 3.2) * adsr(n, 0.005, 0.0, 1.0, 0.05) * 0.22
    if kind == "bass":
        x = np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        return x * adsr(n, 0.02, 0.2, 0.7, 0.3) * 0.32
    raise ValueError(kind)


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= 1.0 / (1.0 + (f / cutoff) ** 4)
    return np.fft.irfft(spec, len(x))


def echo(x: np.ndarray, delay_s: float, fb: float, mix: float) -> np.ndarray:
    d = int(delay_s * SR)
    out = x.copy()
    for k in range(1, 5):
        if k * d >= len(x):
            break
        out[k * d:] += x[:-k * d] * (mix * fb ** (k - 1))
    return out


def make_track(root: int, minor: bool, bpm: float, seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = 4 * beat
    scale = [0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]
    prog = [0, 5, 2, 6] if minor else [0, 4, 5, 3]        # scale degrees of the chord roots
    n_total = int(seconds * SR)
    L = np.zeros(n_total)
    R = np.zeros(n_total)

    def put(buf, sig, start_s):
        i = int(start_s * SR)
        j = min(n_total, i + len(sig))
        if i < n_total:
            buf[i:j] += sig[:j - i]

    n_bars = int(seconds / bar) + 1
    for b in range(n_bars):
        deg = prog[b % len(prog)]
        notes = [root + scale[(deg + k) % 7] + 12 * ((deg + k) // 7) for k in (0, 2, 4)]
        t0 = b * bar
        for n in notes:                                          # slow pad
            pad = tone(midi_hz(n + 12), bar + 1.0, "pad", rng)
            put(L, pad * 1.0, t0)
            put(R, pad * 0.9, t0)
        put(L, tone(midi_hz(notes[0] - 12), bar, "bass", rng), t0)
        put(R, tone(midi_hz(notes[0] - 12), bar, "bass", rng), t0)
        for s in range(8):                                       # soft eighth-note arpeggio
            n = notes[[0, 1, 2, 1, 0, 2, 1, 2][s]] + 24
            pl = tone(midi_hz(n), beat * 1.2, "pluck", rng)
            pan = 0.5 + 0.35 * np.sin(s + b)
            put(L, pl * (1 - pan) * 1.4, t0 + s * beat / 2)
            put(R, pl * pan * 1.4, t0 + s * beat / 2)
    L, R = lowpass(echo(L, beat * 0.75, 0.5, 0.35), 5200), lowpass(echo(R, beat * 0.5, 0.5, 0.35), 5200)
    fade = int(3 * SR)
    for x in (L, R):
        x[:fade] *= np.linspace(0, 1, fade)
        x[-fade:] *= np.linspace(1, 0, fade)
    st = np.stack([L, R], axis=1)
    st /= max(1e-9, np.abs(st).max()) / 0.7
    return st.astype(np.float32)


TRACKS = [
    ("calm_horizon", 57, True, 84, 11),
    ("city_lights", 60, False, 96, 22),
    ("night_drive", 55, True, 76, 33),
    ("open_road", 62, False, 104, 44),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=150.0)
    a = ap.parse_args()
    out_dir = ROOT / "music"
    out_dir.mkdir(exist_ok=True)
    for name, root, minor, bpm, seed in TRACKS:
        pcm = make_track(root, minor, bpm, a.seconds, seed)
        raw = ROOT / ".tmp" / f"{name}.f32"
        raw.parent.mkdir(exist_ok=True)
        pcm.tofile(raw)
        dst = out_dir / f"{name}.m4a"
        subprocess.run([ffmpeg.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR),
                        "-ac", "2", "-i", str(raw), "-c:a", "aac", "-b:a", "160k", str(dst)],
                       check=True, env=subprocess_env())
        raw.unlink(missing_ok=True)
        print(f"{dst.name}: {a.seconds:.0f}s, {dst.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
