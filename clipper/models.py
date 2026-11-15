"""Shared data contracts between pipeline stages.

TIME BASE: unless a field says otherwise, times are *stream seconds* — seconds since the
start of chunk 0 of the current stream (chunk.start_s + offset inside the chunk). Fields
named `t` inside per-chunk analysis results are *chunk-relative* seconds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

Track = Literal["gameplay", "cutscene"]
Route = Literal["drop", "backup", "publish"]
FrameKind = Literal["gameplay", "cutscene", "menu", "loading", "map", "other"]

# HUD event vocabulary (Claude grid schema enum uses exactly these)
HUD_EVENTS = ("wanted_up", "wanted_stars", "wasted", "busted", "mission_passed",
              "mission_failed", "notification", "other")


# --------------------------------------------------------------------- recording
@dataclass
class Chunk:
    idx: int
    path: str
    start_s: float = 0.0          # stream seconds where this chunk begins
    duration_s: float = 0.0
    status: str = "recorded"      # recorded | processing | processed | deleted
    recorded_at: float = 0.0      # unix time the chunk was closed
    stream_id: str = ""

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


# --------------------------------------------------------------------- analysis
@dataclass
class AudioFeatures:
    """Per-hop audio features for one chunk (CPU signal processing)."""
    chunk_idx: int
    hop_s: float
    t: np.ndarray                 # chunk-relative hop centres (s)
    loudness_db: np.ndarray       # RMS dBFS
    loud_z: np.ndarray            # z-score vs rolling stream baseline
    spike: np.ndarray             # 0..1 sudden jump above recent loudness
    transient: np.ndarray         # 0..1 sharp bursts (gunfire, explosions, crashes)
    music: np.ndarray             # bool: steady tonal sound (radio / score)
    swell: np.ndarray             # 0..1 rising music level
    radio_ranges: list[tuple[float, float]] = field(default_factory=list)  # chunk-relative

    def at(self, t0: float, t1: float) -> slice:
        """Index slice covering chunk-relative [t0, t1)."""
        i0 = int(max(0, np.searchsorted(self.t, t0)))
        i1 = int(max(i0, np.searchsorted(self.t, t1)))
        return slice(i0, i1)


@dataclass
class VisualFrame:
    """One sampled frame (every 3 s). AI fields from the Claude grid call, CV fields measured."""
    t: float                                  # chunk-relative
    kind: str = "gameplay"                    # FrameKind
    hud_events: list[str] = field(default_factory=list)
    wanted_stars: int = 0
    title_text: str | None = None
    action: int = 0                           # 0..10, Claude's on-screen action rating
    scene_tag: str | None = None              # short scene / location label for newness+repeats
    letterbox: bool = False                   # CV cross-check
    from_ai: bool = False                     # False if Claude was unavailable


@dataclass
class MotionSeries:
    """Dense low-res motion measurements (motion_fps) for one chunk, all 0..1."""
    t: np.ndarray                 # chunk-relative
    motion: np.ndarray            # mean abs frame difference
    scene_change: np.ndarray      # hard-cut likelihood
    flash: np.ndarray             # brightness jump
    shake: np.ndarray             # global shift magnitude
    letterbox: np.ndarray         # bool per sample
    static: np.ndarray            # bool: near-zero motion (menus/loading/pause)


@dataclass
class VisualFeatures:
    chunk_idx: int
    frames: list[VisualFrame]
    motion: MotionSeries
    frame_paths: dict[float, str] = field(default_factory=dict)   # t -> small jpg path
    grid_paths: list[str] = field(default_factory=list)
    ai_ok: bool = True


# --------------------------------------------------------------------- scoring
@dataclass
class ClipMeta:
    title: str = ""
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    hook: str = ""
    spoiler: bool = False
    source: str = "template"      # judge | template


@dataclass
class Candidate:
    id: str                       # f"{stream_id}-{chunk_idx}-{int(peak_s*10)}"
    chunk_idx: int                # chunk containing the peak
    start_s: float                # stream seconds
    peak_s: float
    end_s: float
    track: str                    # Track
    signals: dict[str, float] = field(default_factory=dict)     # signal name -> 0..1
    penalties: dict[str, float] = field(default_factory=dict)   # penalty name -> points
    signal_score: float = 0.0     # 0..100
    judge: dict[str, Any] | None = None   # raw judge JSON (rating, ...)
    final_score: float = 0.0      # 0..100
    route: str = "drop"           # Route
    events: list[str] = field(default_factory=list)  # human readable detected events
    scene_tag: str | None = None
    radio_ranges: list[tuple[float, float]] = field(default_factory=list)  # stream seconds
    meta: ClipMeta = field(default_factory=ClipMeta)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        d = dict(d)
        d["meta"] = ClipMeta(**(d.get("meta") or {}))
        d["radio_ranges"] = [tuple(r) for r in d.get("radio_ranges", [])]
        return cls(**d)


# --------------------------------------------------------------------- production / publishing
@dataclass
class ClipSidecar:
    """JSON written next to every finished clip (<clip>.json)."""
    clip_id: str
    clip_path: str
    track: str
    route: str
    final_score: float
    signal_score: float
    title: str
    description: str
    hashtags: list[str]
    hook: str
    spoiler: bool
    duration_s: float
    start_s: float
    end_s: float
    music_track: str | None = None
    created_at: float = 0.0
    candidate: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueueItem:
    clip_id: str
    clip_path: str
    sidecar_path: str
    final_score: float
    track: str
    kind: str                     # publish | backup
    status: str = "queued"        # queued | scheduled | uploaded | failed | skipped
    publish_at: str | None = None # RFC3339 UTC when scheduled
    video_id: str | None = None
    attempts: int = 0
    created_at: float = 0.0
    error: str | None = None
