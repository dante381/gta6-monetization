"""Publish queue: finished clips (ClipSidecar) become QueueItems in the state DB."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..models import ClipSidecar, QueueItem


def sidecar_path_for(clip_path: str | Path) -> Path:
    return Path(clip_path).with_suffix(".json")


def load_sidecar(path: str | Path) -> ClipSidecar:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClipSidecar(**d)


def enqueue(sidecar: ClipSidecar, state: Any, cfg: Any = None,
            sidecar_path: str | Path | None = None) -> QueueItem | None:
    """Add a produced clip to the publish queue.

    route publish -> kind publish (scheduled post); route backup -> kind backup (unlisted).
    Anything else (drop) is not queued and returns None. Re-enqueueing the same clip_id is a
    no-op (the existing queue row wins).
    """
    if sidecar.route == "publish":
        kind = "publish"
    elif sidecar.route == "backup":
        kind = "backup"
    else:
        return None
    item = QueueItem(
        clip_id=sidecar.clip_id,
        clip_path=str(sidecar.clip_path),
        sidecar_path=str(sidecar_path or sidecar_path_for(sidecar.clip_path)),
        final_score=float(sidecar.final_score),
        track=sidecar.track,
        kind=kind,
        status="queued",
        created_at=sidecar.created_at or time.time(),
    )
    state.queue_add(item)
    return state.queue_get(item.clip_id) or item
