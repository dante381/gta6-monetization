"""Royalty-free music from paths.MUSIC: deterministic track + offset per candidate."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from .. import ffmpeg, paths

log = logging.getLogger(__name__)

MUSIC_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus")
_warned_empty = False


@dataclass
class MusicPick:
    path: Path
    offset_s: float        # seek into the track
    track_s: float         # track length
    loop: bool             # track shorter than needed -> loop it

    @property
    def name(self) -> str:
        return self.path.name


def list_tracks(music_dir: Path | None = None) -> list[Path]:
    d = Path(music_dir or paths.MUSIC)
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.is_file() and p.suffix.lower() in MUSIC_EXTS),
                  key=lambda p: p.name.lower())


def _h(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def pick_track(cand_id: str, need_s: float, music_dir: Path | None = None) -> MusicPick | None:
    """Same candidate id -> same track and offset. None (warns once) when the folder is empty."""
    global _warned_empty
    tracks = list_tracks(music_dir)
    if not tracks:
        if not _warned_empty:
            log.warning("no music files in %s: radio sections will just be muted", music_dir or paths.MUSIC)
            _warned_empty = True
        return None
    path = tracks[_h(cand_id) % len(tracks)]
    try:
        track_s = ffmpeg.duration(path)
    except ffmpeg.FfmpegError as e:
        log.warning("cannot probe music %s (%s); using offset 0", path.name, e)
        track_s = 0.0
    spare = track_s - need_s
    if track_s <= 0 or spare <= 0.5:
        return MusicPick(path, 0.0, track_s, loop=True)
    frac = (_h(cand_id + ":offset") % 10_000) / 10_000
    return MusicPick(path, round(frac * spare, 2), track_s, loop=False)
