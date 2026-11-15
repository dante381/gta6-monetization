"""Crash-safe state (SQLite, WAL). One connection per process; safe across recorder/worker/supervisor.

This is the spec's "state file": which chunks are processed, candidates, the publish queue,
upload counts, seen scenes, hold flags and repair attempts.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .models import Candidate, Chunk, QueueItem

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    stream_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    path TEXT NOT NULL,
    start_s REAL DEFAULT 0,
    duration_s REAL DEFAULT 0,
    status TEXT DEFAULT 'recorded',
    recorded_at REAL DEFAULT 0,
    processed_at REAL,
    clips_cut INTEGER DEFAULT 0,
    PRIMARY KEY (stream_id, idx)
);
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    stream_id TEXT,
    chunk_idx INTEGER,
    peak_s REAL,
    track TEXT,
    final_score REAL,
    route TEXT,
    scene_tag TEXT,
    produced INTEGER DEFAULT 0,
    json TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS queue (
    clip_id TEXT PRIMARY KEY,
    clip_path TEXT,
    sidecar_path TEXT,
    final_score REAL,
    track TEXT,
    kind TEXT,
    status TEXT,
    publish_at TEXT,
    video_id TEXT,
    attempts INTEGER DEFAULT 0,
    created_at REAL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS upload_log (
    clip_id TEXT,
    day TEXT,              -- Pacific-time date of the API call (YYYY-MM-DD)
    ts REAL,
    kind TEXT,
    video_id TEXT
);
CREATE TABLE IF NOT EXISTS seen_scenes (
    tag TEXT PRIMARY KEY,
    first_s REAL,
    count INTEGER DEFAULT 0
);
"""

_QUEUE_COLS = [f.name for f in fields(QueueItem)]


class StateDB:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or paths.DATA.state_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None,
                                    check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def _x(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(args))

    # ------------------------------------------------------------------ kv / flags
    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self._x("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def kv_set(self, key: str, value: Any) -> None:
        self._x("INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)))

    # Hold flags set by the supervisor during repair
    def publish_hold(self) -> bool:
        return bool(self.kv_get("hold.publish", False))

    def set_publish_hold(self, on: bool, reason: str = "") -> None:
        self.kv_set("hold.publish", on)
        self.kv_set("hold.publish_reason", reason)

    def keep_chunks(self) -> bool:
        return bool(self.kv_get("hold.keep_chunks", False))

    def set_keep_chunks(self, on: bool) -> None:
        self.kv_set("hold.keep_chunks", on)

    # ------------------------------------------------------------------ stream
    def current_stream(self) -> str | None:
        return self.kv_get("stream.current")

    def set_current_stream(self, stream_id: str) -> None:
        self.kv_set("stream.current", stream_id)

    # ------------------------------------------------------------------ chunks
    def add_chunk(self, c: Chunk) -> None:
        self._x("""INSERT INTO chunks(stream_id, idx, path, start_s, duration_s, status, recorded_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(stream_id, idx) DO UPDATE SET path=excluded.path,
                     start_s=excluded.start_s, duration_s=excluded.duration_s,
                     recorded_at=excluded.recorded_at""",
                (c.stream_id, c.idx, c.path, c.start_s, c.duration_s, c.status, c.recorded_at or time.time()))

    def update_chunk(self, stream_id: str, idx: int, **f: Any) -> None:
        if not f:
            return
        cols = ", ".join(f"{k}=?" for k in f)
        self._x(f"UPDATE chunks SET {cols} WHERE stream_id=? AND idx=?", [*f.values(), stream_id, idx])

    @staticmethod
    def _chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(idx=row["idx"], path=row["path"], start_s=row["start_s"], duration_s=row["duration_s"],
                     status=row["status"], recorded_at=row["recorded_at"], stream_id=row["stream_id"])

    def get_chunk(self, stream_id: str, idx: int) -> Chunk | None:
        row = self._x("SELECT * FROM chunks WHERE stream_id=? AND idx=?", (stream_id, idx)).fetchone()
        return self._chunk(row) if row else None

    def chunks(self, stream_id: str, status: str | None = None) -> list[Chunk]:
        if status:
            rows = self._x("SELECT * FROM chunks WHERE stream_id=? AND status=? ORDER BY idx", (stream_id, status))
        else:
            rows = self._x("SELECT * FROM chunks WHERE stream_id=? ORDER BY idx", (stream_id,))
        return [self._chunk(r) for r in rows]

    def last_chunk_idx(self, stream_id: str) -> int:
        row = self._x("SELECT MAX(idx) AS m FROM chunks WHERE stream_id=?", (stream_id,)).fetchone()
        return -1 if row["m"] is None else int(row["m"])

    def next_chunk_to_process(self, stream_id: str) -> Chunk | None:
        """Oldest chunk not yet processed (a 'processing' chunk left by a crash is retried)."""
        row = self._x("""SELECT * FROM chunks WHERE stream_id=? AND status IN ('recorded','processing')
                         ORDER BY idx LIMIT 1""", (stream_id,)).fetchone()
        return self._chunk(row) if row else None

    def chunks_covering(self, stream_id: str, start_s: float, end_s: float) -> list[Chunk]:
        rows = self._x("""SELECT * FROM chunks WHERE stream_id=? AND status!='deleted'
                          AND start_s < ? AND start_s + duration_s > ? ORDER BY idx""",
                       (stream_id, end_s, start_s))
        return [self._chunk(r) for r in rows]

    def mark_clips_cut(self, stream_id: str, idx: int) -> None:
        self.update_chunk(stream_id, idx, clips_cut=1)

    def clips_cut(self, stream_id: str, idx: int) -> bool:
        row = self._x("SELECT clips_cut FROM chunks WHERE stream_id=? AND idx=?", (stream_id, idx)).fetchone()
        return bool(row and row["clips_cut"])

    # ------------------------------------------------------------------ candidates
    def save_candidate(self, c: Candidate, stream_id: str, produced: bool = False) -> None:
        self._x("""INSERT INTO candidates(id, stream_id, chunk_idx, peak_s, track, final_score, route,
                       scene_tag, produced, json, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET final_score=excluded.final_score, route=excluded.route,
                       produced=MAX(candidates.produced, excluded.produced), json=excluded.json,
                       scene_tag=excluded.scene_tag, track=excluded.track""",
                (c.id, stream_id, c.chunk_idx, c.peak_s, c.track, c.final_score, c.route, c.scene_tag,
                 int(produced), json.dumps(c.to_dict()), time.time()))

    def mark_produced(self, cand_id: str) -> None:
        self._x("UPDATE candidates SET produced=1 WHERE id=?", (cand_id,))

    def candidates(self, stream_id: str | None = None, route: str | None = None,
                   produced: bool | None = None) -> list[Candidate]:
        sql, args = "SELECT json FROM candidates WHERE 1=1", []
        if stream_id is not None:
            sql += " AND stream_id=?"; args.append(stream_id)
        if route is not None:
            sql += " AND route=?"; args.append(route)
        if produced is not None:
            sql += " AND produced=?"; args.append(int(produced))
        sql += " ORDER BY peak_s"
        return [Candidate.from_dict(json.loads(r["json"])) for r in self._x(sql, args)]

    def selected_near(self, stream_id: str, t: float, window_s: float) -> list[Candidate]:
        """Already-selected (non-dropped) candidates whose peak is within window_s of t."""
        rows = self._x("""SELECT json FROM candidates WHERE stream_id=? AND route!='drop'
                          AND ABS(peak_s - ?) <= ?""", (stream_id, t, window_s))
        return [Candidate.from_dict(json.loads(r["json"])) for r in rows]

    def selected_scene_tags(self, stream_id: str) -> set[str]:
        rows = self._x("SELECT DISTINCT scene_tag FROM candidates WHERE stream_id=? AND route!='drop' "
                       "AND scene_tag IS NOT NULL", (stream_id,))
        return {r["scene_tag"] for r in rows}

    # ------------------------------------------------------------------ seen scenes (newness)
    def scene_count(self, tag: str) -> int:
        row = self._x("SELECT count FROM seen_scenes WHERE tag=?", (tag,)).fetchone()
        return int(row["count"]) if row else 0

    def mark_scene(self, tag: str, t: float) -> None:
        self._x("""INSERT INTO seen_scenes(tag, first_s, count) VALUES(?,?,1)
                   ON CONFLICT(tag) DO UPDATE SET count=count+1""", (tag, t))

    # ------------------------------------------------------------------ queue
    def queue_add(self, q: QueueItem) -> None:
        d = asdict(q)
        d["created_at"] = d["created_at"] or time.time()
        cols = ", ".join(_QUEUE_COLS)
        marks = ", ".join("?" for _ in _QUEUE_COLS)
        self._x(f"INSERT OR IGNORE INTO queue({cols}) VALUES({marks})", [d[c] for c in _QUEUE_COLS])

    def queue_update(self, clip_id: str, **f: Any) -> None:
        if not f:
            return
        cols = ", ".join(f"{k}=?" for k in f)
        self._x(f"UPDATE queue SET {cols} WHERE clip_id=?", [*f.values(), clip_id])

    def queue_items(self, status: str | None = None, kind: str | None = None) -> list[QueueItem]:
        sql, args = "SELECT * FROM queue WHERE 1=1", []
        if status is not None:
            sql += " AND status=?"; args.append(status)
        if kind is not None:
            sql += " AND kind=?"; args.append(kind)
        sql += " ORDER BY final_score DESC, created_at"
        return [QueueItem(**{c: r[c] for c in _QUEUE_COLS}) for r in self._x(sql, args)]

    def queue_get(self, clip_id: str) -> QueueItem | None:
        r = self._x("SELECT * FROM queue WHERE clip_id=?", (clip_id,)).fetchone()
        return QueueItem(**{c: r[c] for c in _QUEUE_COLS}) if r else None

    # ------------------------------------------------------------------ upload accounting
    def record_upload(self, clip_id: str, day: str, kind: str, video_id: str | None) -> None:
        self._x("INSERT INTO upload_log(clip_id, day, ts, kind, video_id) VALUES(?,?,?,?,?)",
                (clip_id, day, time.time(), kind, video_id))

    def uploads_on(self, day: str) -> int:
        return int(self._x("SELECT COUNT(*) AS n FROM upload_log WHERE day=?", (day,)).fetchone()["n"])
