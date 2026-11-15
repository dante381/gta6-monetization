"""Synthetic AudioFeatures / VisualFeatures builders for the scoring tests (plus sanity checks)."""
from __future__ import annotations

import numpy as np

from clipper.models import AudioFeatures, Chunk, MotionSeries, VisualFeatures, VisualFrame


class Synth:
    """Quiet gameplay baseline; methods paint moments over chunk-relative [t0, t1)."""

    def __init__(self, duration: float = 300.0, *, ai_ok: bool = True, idx: int = 0, base_action: int = 2,
                 tag: str = "city street", hop: float = 0.5, mfps: float = 2.0, frame_dt: float = 3.0):
        self.duration, self.ai_ok, self.idx = duration, ai_ok, idx
        self.at = np.arange(0, duration, hop) + hop / 2
        self.hop = hop
        n = len(self.at)
        self.loud = np.full(n, -35.0)
        self.loud_z = np.zeros(n)
        self.spike = np.full(n, 0.02)
        self.transient = np.full(n, 0.02)
        self.music = np.zeros(n, dtype=bool)
        self.swell = np.zeros(n)
        self.radio: list[tuple[float, float]] = []
        self.mt = np.arange(0, duration, 1 / mfps)
        m = len(self.mt)
        self.motion = np.full(m, 0.08)
        self.scene_change = np.zeros(m)
        self.flash = np.zeros(m)
        self.shake = np.zeros(m)
        self.letter = np.zeros(m, dtype=bool)
        self.static = np.zeros(m, dtype=bool)
        self.frames = [VisualFrame(t=float(t), kind="gameplay", action=base_action if ai_ok else 0,
                                   scene_tag=tag if ai_ok else None, from_ai=ai_ok)
                       for t in np.arange(0, duration, frame_dt)]

    def _a(self, t0, t1):
        return (self.at >= t0) & (self.at < t1)

    def _m(self, t0, t1):
        return (self.mt >= t0) & (self.mt < t1)

    def _frames(self, t0, t1):
        return [f for f in self.frames if t0 <= f.t < t1]

    def set_frames(self, t0, t1, **fields):
        for f in self._frames(t0, t1):
            for k, v in fields.items():
                setattr(f, k, v)
        return self

    def gunfire(self, t0, t1, action: int = 8):
        a = self._a(t0, t1)
        self.spike[a] = 0.95
        on = a & (np.sin(2 * np.pi * 1.0 * self.at) > 0)   # alternating bursts
        self.transient[a] = 0.1
        self.transient[on] = 0.95
        self.loud[a], self.loud_z[a] = -8.0, 4.0
        m = self._m(t0, t1)
        self.motion[m], self.shake[m], self.flash[m] = 0.6, 0.5, 0.7
        if self.ai_ok:
            self.set_frames(t0, t1, action=action)
        return self

    def bang(self, t):
        a = self._a(t, t + 0.6)
        self.spike[a] = 1.0
        self.transient[a] = 1.0
        return self

    def letterbox(self, t0, t1, motion: float = 0.2):
        m = self._m(t0, t1)
        self.letter[m] = True
        self.motion[m] = motion
        self.set_frames(t0, t1, letterbox=True)
        if self.ai_ok:
            self.set_frames(t0, t1, kind="cutscene", action=3)
        return self

    def swell_up(self, t0, t1):
        a = self._a(t0, t1)
        self.swell[a] = np.linspace(0.1, 0.9, a.sum())
        self.music[a] = True
        return self

    def loading(self, t0, t1):
        m = self._m(t0, t1)
        self.static[m] = True
        self.motion[m] = 0.0
        self.set_frames(t0, t1, kind="loading", action=0, scene_tag=None)
        return self

    def radio_on(self, t0, t1):
        self.radio.append((float(t0), float(t1)))
        self.music[self._a(t0, t1)] = True
        return self

    def build(self) -> tuple[AudioFeatures, VisualFeatures]:
        audio = AudioFeatures(chunk_idx=self.idx, hop_s=self.hop, t=self.at.copy(), loudness_db=self.loud.copy(),
                              loud_z=self.loud_z.copy(), spike=self.spike.copy(), transient=self.transient.copy(),
                              music=self.music.copy(), swell=self.swell.copy(), radio_ranges=list(self.radio))
        motion = MotionSeries(t=self.mt.copy(), motion=self.motion.copy(), scene_change=self.scene_change.copy(),
                              flash=self.flash.copy(), shake=self.shake.copy(), letterbox=self.letter.copy(),
                              static=self.static.copy())
        visual = VisualFeatures(chunk_idx=self.idx, frames=self.frames, motion=motion, ai_ok=self.ai_ok)
        return audio, visual


def make_chunk(idx: int = 0, start_s: float = 0.0, duration: float = 300.0, path: str = "",
               stream_id: str = "test-stream") -> Chunk:
    return Chunk(idx=idx, path=path, start_s=start_s, duration_s=duration, stream_id=stream_id)


def cfg_with(**scoring) -> "object":
    from clipper.config import load_config
    return load_config(overrides={"scoring": scoring} if scoring else None)


# ------------------------------------------------------------------ sanity checks of the builder
def test_synth_shapes():
    a, v = Synth(60).gunfire(20, 25).build()
    assert len(a.t) == len(a.spike) == 120
    assert len(v.motion.t) == 120 and len(v.frames) == 20
    assert a.spike[a.at(20, 25)].max() > 0.9
