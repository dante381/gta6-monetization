"""Audio analysis (CPU signal processing, no AI).

ffmpeg decodes the chunk's audio to a 16 kHz mono float32 pipe, read in `audio_block_s` blocks
so memory stays small. Every hop (`audio_hop_s`, 0.5 s) gets a few raw measurements:

    loudness_db  RMS level in dBFS
    flux         normalised positive spectral flux (how much of the spectrum is new)
    flatness     energy-weighted spectral flatness (~0 = pure tones, ~0.5 = white noise)
    stability    cosine similarity of consecutive frame spectra (1 = unchanging spectrum)
    onset_db     largest rise of the 16 ms envelope over the previous ~100 ms (dB)
    burst_db     envelope spread inside the hop, p90 - p10 (dB)
    crest_db     peak / RMS of the samples in the hop (dB)

From these, `derive_features` builds the AudioFeatures outputs:

    loud_z     z-score of loudness against the rolling stream baseline (median / MAD of the
               last `baseline_minutes`, kept across chunks in state.kv["audio.baseline"]).
    spike      0..1, sudden jump above the previous ~10 s *and* above the stream baseline.
               Tonal (music-like) jumps count less than broadband ones.
    transient  0..1, sharp broadband bursts: gunfire, explosions, crashes.
    music      bool, steady tonal sound sustained >= `radio_min_s` (radio / score).
    swell      0..1, rising level inside music: rise vs the quietest point 5-15 s earlier,
               weighted by how far above the stream baseline the music is.
    radio_ranges  chunk-relative (start, end) seconds of the music runs.

Silence (loading screens) is clamped to `audio_silence_db`, so it never yields spikes from
division noise and never enters the baseline.
"""
from __future__ import annotations

import collections
import json
import logging
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..ffmpeg import FFMPEG, FfmpegError, kill_tree, popen
from ..models import AudioFeatures, Chunk

log = logging.getLogger(__name__)

KV_KEY = "audio.baseline"
_EPS = 1e-12

# Defaults for every tunable; cfg.analysis.<name> overrides (settings.yaml).
DEFAULTS: dict[str, Any] = {
    "audio_rate": 16000,
    "audio_hop_s": 0.5,
    "audio_block_s": 10,
    "baseline_minutes": 10,
    "radio_min_s": 20,
    "audio_frame": 1024,              # FFT frame (samples)
    "audio_silence_db": -60.0,        # below this a hop counts as silence
    "audio_mad_floor_db": 3.0,        # minimum baseline spread for loud_z
    "audio_spike_ref_s": 10.0,        # "recent loudness" window before each hop
    "audio_spike_min_db": 6.0,        # jump over recent level where spike starts
    "audio_spike_range_db": 18.0,     # extra jump for spike = 1
    "audio_spike_above_db": 6.0,      # dB above baseline median for full spike
    "audio_tonal_spike_weight": 0.4,  # spike weight for a purely tonal (music) jump
    "audio_onset_min_db": 9.0,        # envelope rise where transient starts
    "audio_onset_range_db": 18.0,     # extra rise for transient = 1
    "audio_flat_tonal": 0.01,         # flatness at/below = fully tonal
    "audio_flat_noisy": 0.2,          # flatness at/above = fully broadband
    "audio_transient_gate_db": 6.0,   # transient gate: 0 at baseline median - this, 1 at the median
    "audio_music_flatness_max": 0.05,
    "audio_music_stability_min": 0.9,
    "audio_music_onset_max_db": 9.0,
    "audio_music_level_dev_db": 6.0,  # |level - local median| must stay below this
    "audio_music_gap_s": 2.0,         # non-tonal gaps this short are bridged
    "audio_swell_rise_db": 3.0,       # rise vs 5-15 s earlier for full swell
    "audio_swell_level_db": 12.0,     # dB above baseline median for full swell
    "audio_baseline_res_s": 1.0,      # resolution of the stored baseline ring
}


def _p(cfg, name: str):
    """Tunable from cfg.analysis with module default fallback."""
    a = None
    try:
        a = cfg.analysis if cfg is not None else None
    except AttributeError:
        a = None
    if a is not None and name in a and a[name] is not None:
        return a[name]
    return DEFAULTS[name]


# ============================================================== per-hop raw measurements
@dataclass
class RawHops:
    hop_s: float
    loudness_db: np.ndarray
    flux: np.ndarray
    flatness: np.ndarray
    stability: np.ndarray
    onset_db: np.ndarray
    burst_db: np.ndarray
    crest_db: np.ndarray

    @property
    def n(self) -> int:
        return len(self.loudness_db)

    @property
    def t(self) -> np.ndarray:
        return (np.arange(self.n) + 0.5) * self.hop_s


class HopExtractor:
    """Streaming per-hop feature extractor. push() any number of samples, then finish()."""

    N_SUB = 32          # envelope sub-frames per hop
    ONSET_LOOKBACK = 6  # sub-frames (~94 ms at 0.5 s hops)

    def __init__(self, rate: int, cfg=None):
        self.rate = int(rate)
        self.hop_s = float(_p(cfg, "audio_hop_s"))
        self.hop_n = int(round(self.hop_s * self.rate))
        self.frame = int(min(_p(cfg, "audio_frame"), self.hop_n))
        self.step = self.frame // 2
        self.n_frames = (self.hop_n - self.frame) // self.step + 1
        self.sub_n = self.hop_n // self.N_SUB
        self.silence_db = float(_p(cfg, "audio_silence_db"))
        self.window = np.hanning(self.frame).astype(np.float32)
        self._fidx = (np.arange(self.n_frames)[:, None] * self.step + np.arange(self.frame)[None, :])
        lo = max(1, int(round(60.0 * self.frame / self.rate)))   # skip DC / rumble below ~60 Hz
        self._bins = slice(lo, self.frame // 2 + 1)
        self._buf = np.zeros(0, dtype=np.float32)
        self._prev_mag: np.ndarray | None = None
        self._prev_env = np.zeros(0, dtype=np.float64)
        self._out: dict[str, list[np.ndarray]] = collections.defaultdict(list)

    def push(self, samples: np.ndarray) -> None:
        x = np.asarray(samples, dtype=np.float32).ravel()
        if self._buf.size:
            x = np.concatenate([self._buf, x])
        n = x.size // self.hop_n
        if n:
            self._process(x[: n * self.hop_n].reshape(n, self.hop_n))
        self._buf = x[n * self.hop_n:].copy()

    def finish(self) -> RawHops:
        # A trailing partial hop of at least half a hop is zero-padded and kept.
        if self._buf.size >= self.hop_n // 2:
            pad = np.zeros(self.hop_n, dtype=np.float32)
            pad[: self._buf.size] = self._buf
            rms_n = self._buf.size
            self._process(pad[None, :], valid_n=rms_n)
        self._buf = np.zeros(0, dtype=np.float32)
        cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in self._out.items()}
        empty = np.zeros(0)
        return RawHops(self.hop_s, *(cat.get(k, empty) for k in
                                     ("loudness_db", "flux", "flatness", "stability",
                                      "onset_db", "burst_db", "crest_db")))

    # ------------------------------------------------------------------ internals
    def _process(self, H: np.ndarray, valid_n: int | None = None) -> None:
        n = H.shape[0]
        sil = self.silence_db
        Hv = H if valid_n is None else H[:, :valid_n]
        ms = np.mean(np.square(Hv, dtype=np.float64), axis=1)
        loud = 10.0 * np.log10(ms + _EPS)
        peak = np.max(np.abs(Hv), axis=1).astype(np.float64)
        crest = np.where(loud > sil, 20.0 * np.log10((peak + 1e-9) / (np.sqrt(ms) + 1e-9)), 0.0)

        # ---- spectra: (n, F, frame) -> magnitudes over the useful bins
        frames = H[:, self._fidx]
        frame_db = 10.0 * np.log10(np.mean(np.square(frames, dtype=np.float64), axis=-1) + _EPS)
        mag = np.abs(np.fft.rfft(frames * self.window, axis=-1))[..., self._bins].astype(np.float64)
        del frames
        power = mag * mag
        fe = power.sum(axis=-1)                                        # frame energy (n, F)
        gm = np.exp(np.mean(np.log(power + 1e-10), axis=-1))
        am = np.mean(power, axis=-1) + 1e-10
        flat_f = np.clip(gm / am, 1e-6, 1.0)
        w = fe + 1e-12
        flat = np.exp(np.sum(w * np.log(flat_f), axis=1) / np.sum(w, axis=1))

        # consecutive-frame comparison, carrying the previous frame across hops/blocks
        flat_mag = mag.reshape(n * self.n_frames, -1)
        prev = np.empty_like(flat_mag)
        prev[1:] = flat_mag[:-1]
        prev[0] = self._prev_mag if self._prev_mag is not None else flat_mag[0]
        self._prev_mag = flat_mag[-1].copy()
        num = np.sum(flat_mag * prev, axis=1)
        den = np.sqrt(np.sum(flat_mag ** 2, axis=1) * np.sum(prev ** 2, axis=1)) + 1e-12
        sim = (num / den).reshape(n, self.n_frames)
        y = np.log1p(1000.0 * flat_mag)
        yp = np.log1p(1000.0 * prev)
        flux_f = (np.sum(np.maximum(0.0, y - yp), axis=1) / (np.sum(y, axis=1) + 1e-9)).reshape(n, self.n_frames)
        silent_f = frame_db <= sil
        sim = np.where(silent_f, 0.0, sim)
        flux_f = np.where(silent_f, 0.0, flux_f)
        stab = np.sum(w * sim, axis=1) / np.sum(w, axis=1)
        flux = np.mean(flux_f, axis=1)

        # ---- 16 ms envelope: onsets and burstiness
        sub = H[:, : self.N_SUB * self.sub_n].reshape(n, self.N_SUB, self.sub_n)
        env = np.maximum(10.0 * np.log10(np.mean(np.square(sub, dtype=np.float64), axis=-1) + _EPS), sil)
        if valid_n is not None:
            keep = max(1, valid_n // self.sub_n)
            env[:, keep:] = env[:, keep - 1: keep]
        flat_env = env.ravel()
        lb = self.ONSET_LOOKBACK
        hist = self._prev_env if self._prev_env.size else np.full(lb, flat_env[0])
        ext = np.concatenate([hist[-lb:], flat_env])
        if ext.size < lb + 1:
            ext = np.concatenate([np.full(lb + 1 - ext.size, ext[0]), ext])
        win = np.lib.stride_tricks.sliding_window_view(ext[:-1], lb)[-flat_env.size:]
        rise = (flat_env - win.min(axis=1)).reshape(n, self.N_SUB)
        self._prev_env = flat_env[-lb:].copy()
        onset = rise.max(axis=1)
        p10, p90 = np.percentile(env, [10, 90], axis=1)
        burst = p90 - p10

        for k, v in (("loudness_db", loud), ("flux", flux), ("flatness", flat), ("stability", stab),
                     ("onset_db", onset), ("burst_db", burst), ("crest_db", crest)):
            self._out[k].append(np.asarray(v, dtype=np.float64))


# ============================================================== baseline
@dataclass
class Baseline:
    """Loudness history of the stream before this chunk (non-silent values, oldest first)."""
    history_db: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tail_db: np.ndarray = field(default_factory=lambda: np.zeros(0))   # previous chunk's last hops
    tonal_tail_s: float = 0.0          # tonal run (s) still open at the end of the previous chunk

    def stats(self, own_db: np.ndarray, cfg=None, own_res_s: float = 0.5) -> tuple[float, float]:
        """(median, sigma) over history + this chunk's non-silent hops, last baseline_minutes."""
        sil = float(_p(cfg, "audio_silence_db"))
        res = float(_p(cfg, "audio_baseline_res_s"))
        own = decimate(np.asarray(own_db, dtype=np.float64), own_res_s, res)
        vals = np.concatenate([np.asarray(self.history_db, dtype=np.float64), own])
        vals = vals[vals > sil]
        keep = int(round(float(_p(cfg, "baseline_minutes")) * 60.0 / res))
        vals = vals[-keep:] if keep > 0 else vals
        if vals.size == 0:
            return sil, float(_p(cfg, "audio_mad_floor_db"))
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        return med, max(1.4826 * mad, float(_p(cfg, "audio_mad_floor_db")))


def decimate(db: np.ndarray, src_res_s: float, dst_res_s: float) -> np.ndarray:
    """Average dB values (in the power domain) down to dst_res_s resolution."""
    k = max(1, int(round(dst_res_s / src_res_s)))
    if k == 1 or db.size == 0:
        return db.astype(np.float64)
    m = db.size // k
    head = db[: m * k].reshape(m, k) if m else np.zeros((0, k))
    out = 10.0 * np.log10(np.mean(10.0 ** (head / 10.0), axis=1) + _EPS) if m else np.zeros(0)
    if db.size > m * k:
        rest = db[m * k:]
        out = np.append(out, 10.0 * np.log10(np.mean(10.0 ** (rest / 10.0)) + _EPS))
    return out


def load_baseline(state, chunk: Chunk, cfg=None) -> Baseline:
    """History for `chunk` from state.kv: all stored segments of the same stream with idx < chunk.idx."""
    raw = state.kv_get(KV_KEY) if state is not None else None
    if not isinstance(raw, dict) or raw.get("stream_id") != (chunk.stream_id or ""):
        return Baseline()
    segs = raw.get("segments") or {}
    idxs = sorted(int(k) for k in segs if int(k) < chunk.idx)
    if not idxs:
        return Baseline()
    hist = np.concatenate([np.asarray(segs[str(i)].get("db", []), dtype=np.float64) for i in idxs])
    keep = int(round(float(_p(cfg, "baseline_minutes")) * 60.0 / float(raw.get("res_s", 1.0))))
    hist = hist[-keep:] if keep > 0 else hist
    last = segs[str(idxs[-1])]
    if idxs[-1] == chunk.idx - 1:
        tail = np.asarray(last.get("tail", []), dtype=np.float64)
        tonal = float(last.get("tonal_tail_s", 0.0))
    else:
        tail, tonal = np.zeros(0), 0.0
    return Baseline(history_db=hist, tail_db=tail, tonal_tail_s=tonal)


def save_baseline(state, chunk: Chunk, raw: RawHops, feats: AudioFeatures, cfg=None) -> None:
    """Store this chunk's decimated loudness (non-silent) + tail; keep only what baseline_minutes needs.

    Segments are keyed by chunk idx, so re-analysing a chunk after a crash replaces, never doubles.
    """
    if state is None:
        return
    res = float(_p(cfg, "audio_baseline_res_s"))
    sil = float(_p(cfg, "audio_silence_db"))
    prev = state.kv_get(KV_KEY)
    if not isinstance(prev, dict) or prev.get("stream_id") != (chunk.stream_id or ""):
        prev = {"v": 1, "stream_id": chunk.stream_id or "", "res_s": res, "segments": {}}
    segs: dict[str, Any] = dict(prev.get("segments") or {})
    dec = decimate(raw.loudness_db, raw.hop_s, res)
    dec = dec[dec > sil]
    ref_n = int(round(float(_p(cfg, "audio_spike_ref_s")) / raw.hop_s))
    segs[str(chunk.idx)] = {
        "db": [round(float(v), 1) for v in dec],
        "tail": [round(float(v), 1) for v in raw.loudness_db[-ref_n:]],
        "tonal_tail_s": round(_trailing_tonal_s(feats, raw.hop_s), 2),
    }
    keep = int(round(float(_p(cfg, "baseline_minutes")) * 60.0 / res))
    total = 0
    kept: dict[str, Any] = {}
    for k in sorted(segs, key=int, reverse=True):
        if total >= keep:
            break
        kept[k] = segs[k]
        total += len(segs[k]["db"])
    state.kv_set(KV_KEY, {"v": 1, "stream_id": chunk.stream_id or "", "res_s": res,
                          "chunk_idx": chunk.idx, "segments": kept})


def _trailing_tonal_s(feats: AudioFeatures, hop_s: float) -> float:
    m = np.asarray(feats.music, dtype=bool)
    if m.size == 0 or not m[-1]:
        return 0.0
    i = m.size - 1
    while i >= 0 and m[i]:
        i -= 1
    return float((m.size - 1 - i) * hop_s)


# ============================================================== ranges helpers
def merge_ranges(ranges: Iterable[Sequence[float]], gap_s: float = 0.0) -> list[tuple[float, float]]:
    """Sort and merge overlapping ranges (and ranges separated by <= gap_s)."""
    rs = sorted((float(a), float(b)) for a, b in ranges if float(b) > float(a))
    out: list[tuple[float, float]] = []
    for a, b in rs:
        if out and a <= out[-1][1] + gap_s:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def to_stream_ranges(ranges: Iterable[Sequence[float]], chunk_start_s: float,
                     gap_s: float = 0.0) -> list[tuple[float, float]]:
    """Chunk-relative ranges -> merged stream-second ranges (chunk.start_s + t)."""
    return [(round(chunk_start_s + a, 3), round(chunk_start_s + b, 3)) for a, b in merge_ranges(ranges, gap_s)]


def clip_ranges(ranges: Iterable[Sequence[float]], start_s: float, end_s: float) -> list[tuple[float, float]]:
    """Intersect ranges with [start_s, end_s] (any time base)."""
    out = []
    for a, b in merge_ranges(ranges):
        a2, b2 = max(a, start_s), min(b, end_s)
        if b2 > a2:
            out.append((a2, b2))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(i0, i1_exclusive)] of True runs."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


# ============================================================== derived features
def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _rolling_median(x: np.ndarray, half: int) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    pad = np.pad(x, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1), axis=1)


def _moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    if x.size == 0 or n <= 1:
        return x.copy()
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")


def derive_features(raw: RawHops, cfg=None, baseline: Baseline | None = None,
                    chunk_idx: int = 0) -> AudioFeatures:
    bl = baseline or Baseline()
    hop = raw.hop_s
    n = raw.n
    t = raw.t
    L = raw.loudness_db
    sil = float(_p(cfg, "audio_silence_db"))
    if n == 0:
        z = np.zeros(0)
        return AudioFeatures(chunk_idx, hop, t, L, z, z, z, z.astype(bool), z, [])

    Lc = np.maximum(L, sil)
    med, sigma = bl.stats(L, cfg, hop)
    loud_z = (Lc - med) / sigma

    # broadband (noisy) vs tonal, 0..1 on a log flatness scale
    lf0 = np.log10(float(_p(cfg, "audio_flat_tonal")))
    lf1 = np.log10(float(_p(cfg, "audio_flat_noisy")))
    broadband = _clip01((np.log10(np.maximum(raw.flatness, 1e-6)) - lf0) / (lf1 - lf0))

    # ---- spike: jump over the previous ~10 s, also above the stream baseline
    ref_n = max(1, int(round(float(_p(cfg, "audio_spike_ref_s")) / hop)))
    tail = np.maximum(np.asarray(bl.tail_db, dtype=np.float64)[-ref_n:], sil)
    ext = np.concatenate([tail, Lc])
    off = tail.size
    ref = np.empty(n)
    for i in range(n):
        j = off + i
        w = ext[max(0, j - ref_n): j]
        ref[i] = np.median(w) if w.size else med
    ref = np.maximum(ref, sil)
    jump = Lc - ref
    jump_s = _clip01((jump - float(_p(cfg, "audio_spike_min_db"))) / float(_p(cfg, "audio_spike_range_db")))
    above_s = _clip01((Lc - med) / float(_p(cfg, "audio_spike_above_db")))
    tw = float(_p(cfg, "audio_tonal_spike_weight"))
    spike = jump_s * above_s * (tw + (1.0 - tw) * broadband)
    spike[L <= sil] = 0.0

    # ---- transient: sharp broadband bursts
    onset_s = _clip01((raw.onset_db - float(_p(cfg, "audio_onset_min_db"))) / float(_p(cfg, "audio_onset_range_db")))
    burst_s = _clip01((raw.burst_db - float(_p(cfg, "audio_onset_min_db"))) / float(_p(cfg, "audio_onset_range_db")))
    crest_s = _clip01((raw.crest_db - 6.0) / 6.0)
    gate_db = float(_p(cfg, "audio_transient_gate_db"))
    gate = _clip01((Lc - med + gate_db) / gate_db)          # 0 at med - gate_db, 1 at/above median
    strength = np.maximum(onset_s, burst_s) * (0.7 + 0.3 * crest_s)
    transient = _clip01(gate * broadband * strength)
    transient[L <= sil] = 0.0
    # coming back from silence (loading screen ends) is a restart, not a bang
    prev_sil = np.r_[(bl.tail_db[-1] <= sil) if bl.tail_db.size else False, L[:-1] <= sil]
    transient[prev_sil & (Lc < med + 6.0)] = 0.0

    # ---- music: steady tonal sound sustained >= radio_min_s
    local_med = _rolling_median(Lc, max(1, int(round(5.0 / hop))))
    onset_med = _rolling_median(raw.onset_db, max(1, int(round(1.0 / hop))))   # onset "variance"
    tonal = ((raw.flatness <= float(_p(cfg, "audio_music_flatness_max")))
             & (raw.stability >= float(_p(cfg, "audio_music_stability_min")))
             & (onset_med <= float(_p(cfg, "audio_music_onset_max_db")))
             & (np.abs(Lc - local_med) <= float(_p(cfg, "audio_music_level_dev_db")))
             & (L > sil))
    gap_n = int(round(float(_p(cfg, "audio_music_gap_s")) / hop))
    filled = tonal.copy()
    runs = _runs(tonal)
    for (a0, a1), (b0, _b1) in zip(runs, runs[1:]):
        if b0 - a1 <= gap_n:
            filled[a1:b0] = True
    min_n = int(np.ceil(float(_p(cfg, "radio_min_s")) / hop))
    carry_n = int(round(bl.tonal_tail_s / hop))
    music = np.zeros(n, dtype=bool)
    for a, b in _runs(filled):
        length = b - a + (carry_n if a == 0 else 0)
        if length >= min_n:
            music[a:b] = True
    radio_ranges = [(round(max(0.0, a * hop), 3), round(b * hop, 3)) for a, b in _runs(music)]

    # ---- swell: level rising inside music (vs quietest point 5-15 s earlier), above baseline
    sm_n = max(1, int(round(2.0 / hop)))
    near_n, far_n = int(round(5.0 / hop)), int(round(15.0 / hop))
    rise_db = float(_p(cfg, "audio_swell_rise_db"))
    lvl_db = float(_p(cfg, "audio_swell_level_db"))
    swell = np.zeros(n)
    Ls = Lc.copy()
    for a, b in _runs(music):
        Ls[a:b] = _moving_avg(Lc[a:b], sm_n)      # smooth inside the section only
        for i in range(a, b):
            lo, hi = max(a, i - far_n), max(a, i - near_n)
            rise = Ls[i] - Ls[lo: hi + 1].min()
            recent_max = Ls[max(a, i - near_n): i + 1].max()
            falling = _clip01(1.0 - (recent_max - Ls[i]) / rise_db)
            swell[i] = _clip01(rise / rise_db) * _clip01((Ls[i] - med) / lvl_db) * falling

    return AudioFeatures(chunk_idx=chunk_idx, hop_s=hop, t=t, loudness_db=L, loud_z=loud_z,
                         spike=spike, transient=transient, music=music, swell=swell,
                         radio_ranges=radio_ranges)


def features_from_samples(samples: np.ndarray, rate: int, cfg=None, baseline: Baseline | None = None,
                          chunk_idx: int = 0, return_raw: bool = False):
    """Pure helper: mono float samples -> AudioFeatures (optionally also the RawHops)."""
    ex = HopExtractor(rate, cfg)
    x = np.asarray(samples, dtype=np.float32).ravel()
    blk = int(float(_p(cfg, "audio_block_s")) * rate)
    for i in range(0, x.size, max(blk, ex.hop_n)):
        ex.push(x[i: i + blk])
    raw = ex.finish()
    feats = derive_features(raw, cfg, baseline, chunk_idx)
    return (feats, raw) if return_raw else feats


# ============================================================== decode + entry point
def _drain_stderr(stream, sink: collections.deque) -> None:
    try:
        for part in iter(lambda: stream.read(4096), b""):
            sink.append(part)
    finally:
        stream.close()


def decode_blocks(path: str, rate: int, block_s: float,
                  on_tick: Callable[[], None] | None = None) -> Iterable[np.ndarray]:
    """Yield float32 mono blocks of `block_s` seconds decoded by ffmpeg; raises FfmpegError on failure."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(int(rate)), "-f", "f32le", "pipe:1"]
    proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    err: collections.deque[bytes] = collections.deque(maxlen=64)
    th = threading.Thread(target=_drain_stderr, args=(proc.stderr, err), daemon=True)
    th.start()
    nbytes = int(block_s * rate) * 4
    carry = b""
    ok = False
    try:
        while True:
            data = proc.stdout.read(nbytes)
            if not data:
                break
            data = carry + data
            usable = len(data) // 4 * 4
            carry = data[usable:]
            if usable:
                yield np.frombuffer(data[:usable], dtype="<f4")
            if on_tick:
                on_tick()
        proc.wait(timeout=60)
        ok = True
    finally:
        if not ok and proc.poll() is None:
            kill_tree(proc)
        try:
            proc.stdout.close()
        except Exception:
            pass
        th.join(timeout=5)
    if proc.returncode != 0:
        stderr = b"".join(err).decode("utf-8", "replace")
        raise FfmpegError(f"audio decode failed ({proc.returncode}): {stderr.strip()[-800:]}",
                          proc.returncode, stderr)


def analyze_audio(chunk: Chunk, cfg, state, on_tick: Callable[[], None] | None = None) -> AudioFeatures:
    """Decode the chunk's audio (streamed), compute per-hop features, update the rolling baseline."""
    rate = int(_p(cfg, "audio_rate"))
    ex = HopExtractor(rate, cfg)
    got = 0
    for block in decode_blocks(chunk.path, rate, float(_p(cfg, "audio_block_s")), on_tick):
        ex.push(block)
        got += block.size
    if got == 0:
        raise FfmpegError(f"no audio decoded from {chunk.path}")
    raw = ex.finish()
    baseline = load_baseline(state, chunk, cfg)
    feats = derive_features(raw, cfg, baseline, chunk.idx)
    save_baseline(state, chunk, raw, feats, cfg)
    if on_tick:
        on_tick()
    log.info("audio chunk %s: %d hops, %d music ranges, max spike %.2f, max transient %.2f",
             chunk.idx, raw.n, len(feats.radio_ranges),
             float(feats.spike.max(initial=0.0)), float(feats.transient.max(initial=0.0)))
    return feats


def summary(feats: AudioFeatures) -> dict:
    """Small JSON-able summary (debugging / calibration)."""
    return json.loads(json.dumps({
        "chunk_idx": feats.chunk_idx, "hops": int(len(feats.t)),
        "radio_ranges": feats.radio_ranges,
        "max_spike": float(np.max(feats.spike, initial=0.0)),
        "max_transient": float(np.max(feats.transient, initial=0.0)),
        "max_swell": float(np.max(feats.swell, initial=0.0)),
    }))
