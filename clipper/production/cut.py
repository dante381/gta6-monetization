"""Cut: map a candidate's stream-second window onto 1+ chunk files (concat demuxer for 2+)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import Chunk


@dataclass
class CutSource:
    input_args: list[str]     # e.g. ["-i", path] or ["-f", "concat", "-safe", "0", "-i", list]
    probe_path: str           # a real media file to probe (first chunk)
    seek_s: float             # -ss relative to the source start
    duration_s: float
    concat_list: Path | None = None


def _concat_line(p: str) -> str:
    s = Path(p).resolve().as_posix().replace("'", "'\\''")
    return f"file '{s}'"


def plan_cut(start_s: float, end_s: float, chunks: list[Chunk], work_dir: Path) -> CutSource:
    if not chunks:
        raise ValueError("no chunks cover the clip")
    if end_s <= start_s:
        raise ValueError(f"empty clip window {start_s}..{end_s}")
    chunks = sorted(chunks, key=lambda c: (c.start_s, c.idx))
    # keep only chunks overlapping the window (callers may pass a neighbour or two)
    used = [c for c in chunks if c.end_s > start_s and c.start_s < end_s] or chunks[:1]
    if any(c.duration_s <= 0 for c in used):          # unknown length: keep them all
        used = chunks
    first = used[0]
    seek = max(0.0, start_s - first.start_s)
    dur = end_s - start_s
    if len(used) == 1:
        return CutSource(["-i", str(first.path)], str(first.path), seek, dur)
    work_dir.mkdir(parents=True, exist_ok=True)
    lst = work_dir / "concat.txt"
    lst.write_text("ffconcat version 1.0\n" + "\n".join(_concat_line(c.path) for c in used) + "\n",
                   encoding="utf-8")
    return CutSource(["-f", "concat", "-safe", "0", "-i", str(lst)], str(first.path), seek, dur, lst)
