"""Chat as a scoring signal: "the opinion of the chat".

The collector stores AGGREGATED live chat per bucket (wall-clock epoch seconds). This module maps a
candidate window (stream seconds) to epoch, shifted by `chat.lag_s` (viewers react AFTER the moment)
and padded by `chat.window_pad_s`, and turns it into one 0..1 value:

    value = blend of  rate  (message-rate z-score of the window vs the preceding baseline)
                      hype  (hype density)            signal_blend = {rate, hype, clip}
                      clip  (explicit clip requests)

`value is None` means "no usable chat" (disabled, stream wall-clock unknown, or too few messages so
far): the signal is then DROPPED and the remaining weights rescaled, like ai_ok=False.
A quiet window in a stream that does have chat data is 0.0, not None.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_BLEND = {"rate": 0.5, "hype": 0.3, "clip": 0.2}


@dataclass
class ChatSignal:
    value: float | None
    stats: dict = field(default_factory=dict)
    excerpts: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.value is not None


def chat_cfg(cfg) -> dict:
    return dict(cfg.get("chat", {}) or {})


def _f(c: dict, key: str, default: float) -> float:
    v = c.get(key)
    return float(default if v is None else v)


def _blend(c: dict) -> tuple[float, float, float]:
    b = dict(DEFAULT_BLEND)
    b.update({k: float(v) for k, v in (c.get("signal_blend") or {}).items()})
    s = (b["rate"] + b["hype"] + b["clip"]) or 1.0
    return b["rate"] / s, b["hype"] / s, b["clip"] / s


def wall_start(state, cfg, stream_id: str) -> float | None:
    """Epoch of stream second 0 if chat is usable for this stream, else None."""
    c = chat_cfg(cfg)
    if state is None or not c.get("enabled", True):
        return None
    try:
        ws = state.stream_wall_start(stream_id)
        if ws is None:
            return None
        if int(state.chat_total_msgs(stream_id)) < int(_f(c, "min_msgs_total", 20)):
            return None
    except Exception:  # noqa: BLE001 - chat must never break scoring
        log.exception("chat availability check failed")
        return None
    return float(ws)


def baseline_stats(counts) -> tuple[float, float]:
    """Robust (median, scale) of per-bucket message counts. The scale never collapses to 0, so a
    flat baseline (all equal, or empty) still gives a finite z-score."""
    a = np.asarray(counts, dtype=float)
    if a.size == 0:
        return 0.0, 1.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    return med, max(mad, 0.25 * med, 1.0)


def _grid(buckets: list[dict], t0: float, t1: float, step: int) -> np.ndarray:
    """Per-bucket message counts over [t0, t1) on the bucket grid (missing buckets = 0)."""
    n = max(0, int(np.ceil((t1 - t0) / step)))
    out = np.zeros(n)
    for b in buckets:
        i = int((b["t"] - t0) // step)
        if 0 <= i < n:
            out[i] += b["msgs"]
    return out


def _baseline_counts(state, stream_id: str, t_end: float, c: dict, step: int,
                     extra_end: float | None = None) -> np.ndarray:
    """Per-bucket counts of the `baseline_minutes` before t_end (optionally running on to extra_end),
    starting at the first bucket actually seen, so the silence before chat began is not 'normal'."""
    t0 = t_end - _f(c, "baseline_minutes", 10) * 60.0
    t1 = extra_end if extra_end is not None else t_end
    bks = state.chat_between(stream_id, t0, t1)
    if not bks:
        return np.zeros(0)
    first = min(b["t"] for b in bks)
    return _grid(bks, first, t1, step)


def _sanitize(text: str, max_chars: int) -> str:
    out = "".join(ch if ch.isprintable() else " " for ch in str(text))
    out = " ".join(out.split())
    return out[:max_chars].strip()


def chat_signal(state, cfg, stream_id: str, start_s: float, end_s: float, peak_s: float) -> ChatSignal:
    c = chat_cfg(cfg)
    wall = wall_start(state, cfg, stream_id)
    if wall is None:
        return ChatSignal(None)
    lag = _f(c, "lag_s", 12)
    pad = _f(c, "window_pad_s", 4)
    step = max(1, int(_f(c, "bucket_s", 5)))
    t0 = wall + start_s + lag - pad
    t1 = wall + end_s + lag + pad
    try:
        win = state.chat_between(stream_id, t0, t1)
        base = _baseline_counts(state, stream_id, t0, c, step)
    except Exception:  # noqa: BLE001
        log.exception("chat lookup failed")
        return ChatSignal(None)

    counts = _grid(win, t0, t1, step)
    med, scale = baseline_stats(base)
    # the reaction is a burst: compare the busiest few buckets of the window with normal chat
    top = np.sort(counts)[-3:] if counts.size else np.zeros(0)
    win_rate = float(top.mean()) if top.size else 0.0
    z = (win_rate - med) / scale
    rate_s = float(min(1.0, max(0.0, z / max(0.1, _f(c, "rate_full_z", 3.0)))))
    hype = float(sum(float(b["hype"]) for b in win))
    clip_reqs = int(sum(int(b["clip_reqs"]) for b in win))
    msgs = int(sum(int(b["msgs"]) for b in win))
    hype_s = float(min(1.0, hype / max(1e-6, _f(c, "hype_full", 6.0))))
    clip_s = float(min(1.0, clip_reqs / max(1e-6, _f(c, "clip_req_full", 2))))
    wr, wh, wc = _blend(c)
    value = float(min(1.0, max(0.0, wr * rate_s + wh * hype_s + wc * clip_s)))

    n_ex = int(_f(c, "judge_excerpt_n", 12))
    max_chars = int(_f(c, "sample_max_chars", 80))
    ex: list[str] = []
    for b in sorted(win, key=lambda b: -(b["msgs"] + 3 * b["clip_reqs"])):
        for s in b.get("samples") or []:
            s = _sanitize(s, max_chars)
            if s and s not in ex:
                ex.append(s)
    stats = {"msgs": msgs, "rate_per_bucket": round(win_rate, 2), "baseline_per_bucket": round(med, 2),
             "z": round(float(z), 2), "hype": round(hype, 2), "clip_reqs": clip_reqs,
             "rate_score": round(rate_s, 3), "hype_score": round(hype_s, 3), "clip_score": round(clip_s, 3)}
    return ChatSignal(round(value, 4), stats, ex[:max(0, n_ex)])


def chat_events(sig: ChatSignal) -> list[str]:
    if sig.value is None:
        return []
    ev = []
    if sig.stats.get("rate_score", 0) >= 0.6:
        ev.append(f"chat spike (x{sig.stats.get('rate_per_bucket', 0)} vs {sig.stats.get('baseline_per_bucket', 0)} normal)")
    if sig.stats.get("hype_score", 0) >= 0.6:
        ev.append("chat hyped")
    if sig.stats.get("clip_reqs", 0) >= 1:
        ev.append("chat asked for a clip")
    return ev


def chat_curve(state, cfg, stream_id: str, chunk_start_s: float, n: int) -> np.ndarray | None:
    """Per-second chat reaction (0..1) for the n seconds of a chunk, already shifted back by lag_s
    (a reaction at wall time T belongs to the moment T - lag). None when chat is unavailable."""
    c = chat_cfg(cfg)
    wall = wall_start(state, cfg, stream_id)
    if wall is None:
        return None
    lag = _f(c, "lag_s", 12)
    step = max(1, int(_f(c, "bucket_s", 5)))
    e0 = wall + chunk_start_s + lag
    e1 = e0 + n
    try:
        base = _baseline_counts(state, stream_id, e0, c, step, extra_end=e1)   # robust median shrugs off spikes
        bks = state.chat_between(stream_id, e0 - step, e1 + step)
    except Exception:  # noqa: BLE001
        log.exception("chat curve lookup failed")
        return None
    med, scale = baseline_stats(base)
    wr, wh, wc = _blend(c)
    zfull = max(0.1, _f(c, "rate_full_z", 3.0))
    hfull = max(1e-6, _f(c, "hype_full", 6.0)) * step / 20.0   # hype_full is per ~20 s window
    cfull = max(1e-6, _f(c, "clip_req_full", 2))
    out = np.zeros(n)
    for b in bks:
        v = (wr * min(1.0, max(0.0, (b["msgs"] - med) / scale / zfull))
             + wh * min(1.0, b["hype"] / hfull) + wc * min(1.0, b["clip_reqs"] / cfull))
        lo = int(round(b["t"] - e0))
        for i in range(max(0, lo), min(n, lo + step)):
            out[i] = max(out[i], v)
    return np.clip(out, 0.0, 1.0)
