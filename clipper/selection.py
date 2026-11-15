"""Percentile selection: which kept clips get published.

Every scored candidate (drops included) goes into `score_history`. A kept clip is produced as a
BACKUP first (the footage is deleted soon after); `decide` later promotes it to the publish queue
when its final score is strictly above the `selection.percentile` of the recent population AND at
least `selection.floor`. Promotion is a cheap move of the .mp4/.json from DATA.backup to DATA.ready.

Crash safety: the order is  mp4 move -> sidecar rewrite (tmp + rename) -> old sidecar removed ->
candidate route -> queue row. Every call first repairs any promotion that stopped half way (the mp4
is already in DATA.ready) by re-running the idempotent tail of that sequence.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from . import log as eventlog
from . import paths
from .models import Candidate

log = logging.getLogger(__name__)


def _paths(cid: str) -> tuple[Path, Path, Path, Path]:
    return (paths.DATA.backup / f"{cid}.mp4", paths.DATA.backup / f"{cid}.json",
            paths.DATA.ready / f"{cid}.mp4", paths.DATA.ready / f"{cid}.json")


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _finish(state, cid: str, score: float, threshold: float | None, n: int, announce: bool) -> bool:
    """Idempotent tail of a promotion; the mp4 is already in DATA.ready. Returns True if anything changed."""
    _bmp4, bjson, rmp4, rjson = _paths(cid)
    changed = False
    src = bjson if bjson.exists() else rjson
    if src.exists():
        try:
            d = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = None
        if isinstance(d, dict) and (d.get("route") != "publish" or d.get("clip_path") != str(rmp4)
                                    or src is bjson):
            d["route"] = "publish"
            d["clip_path"] = str(rmp4)
            if isinstance(d.get("candidate"), dict):
                d["candidate"]["route"] = "publish"
            _write_json_atomic(rjson, d)
            changed = True
    if bjson.exists():
        bjson.unlink(missing_ok=True)
        changed = True
    state.set_candidate_route(cid, "publish")
    q = state.queue_get(cid)
    if q is not None and q.status == "queued" and (q.kind != "publish" or q.clip_path != str(rmp4)
                                                  or q.sidecar_path != str(rjson)):
        state.queue_update(cid, kind="publish", clip_path=str(rmp4), sidecar_path=str(rjson))
        changed = True
    if announce:
        eventlog.event("promoted", cand=cid, score=score, threshold=threshold, n=n)
    return changed


def _repair(state, stream_id: str) -> int:
    fixed = 0
    for c in state.candidates(stream_id, produced=True):
        if c.route not in ("backup", "publish"):
            continue
        _bmp4, bjson, rmp4, _rjson = _paths(c.id)
        if not rmp4.exists():
            continue
        if _bmp4.exists():       # a stray backup copy next to a finished ready clip: ready wins
            _bmp4.unlink(missing_ok=True)
        q = state.queue_get(c.id)
        needs = (c.route != "publish" or bjson.exists()
                 or (q is not None and q.status == "queued" and q.kind != "publish"))
        if needs:
            _finish(state, c.id, c.final_score, None, 0, announce=False)
            log.warning("repaired half-done promotion of %s", c.id)
            eventlog.event("promotion_repaired", cand=c.id)
            fixed += 1
    return fixed


def _promote(state, cand: Candidate, threshold: float, n: int) -> bool:
    cid = cand.id
    bmp4, _bjson, rmp4, _rjson = _paths(cid)
    q = state.queue_get(cid)
    if q is not None and q.status != "queued":
        key = f"selection.skipped.{cid}"
        if not state.kv_get(key, False):
            state.kv_set(key, True)
            eventlog.event("promote_skipped", cand=cid, queue_status=q.status, score=cand.final_score)
        return False
    if not bmp4.exists():
        return False
    paths.DATA.ready.mkdir(parents=True, exist_ok=True)
    os.replace(bmp4, rmp4)
    _finish(state, cid, cand.final_score, threshold, n, announce=True)
    return True


def decide(cfg, state, stream_id: str, final: bool = False) -> list[Candidate]:
    """Promote produced backup clips of `stream_id` that beat the percentile cut. Returns the promoted."""
    sel = cfg.locked.selection
    _repair(state, stream_id)
    pop = state.history_scores(int(sel.history))
    n = len(pop)
    held = [c for c in state.candidates(stream_id, route="backup", produced=True)]
    if n < int(sel.min_samples) and not final:
        return []
    if final and n < int(sel.end_min_samples):
        eventlog.event("selection", reason="too_few_samples", n=n, final=True, candidates=len(held))
        return []
    threshold = float(np.percentile(pop, float(sel.percentile)))
    floor = float(sel.floor)
    promoted: list[Candidate] = []
    for c in sorted(held, key=lambda c: -c.final_score):
        if c.final_score > threshold and c.final_score >= floor:
            if _promote(state, c, threshold, n):
                c.route = "publish"
                promoted.append(c)
    summary = [round(threshold, 2), n, len(held) - len(promoted)]
    if promoted or final or state.kv_get("selection.last") != summary:   # not every idle loop
        state.kv_set("selection.last", summary)
        eventlog.event("selection", threshold=round(threshold, 2), n=n, promoted=len(promoted),
                       held=len(held) - len(promoted), final=final)
    return promoted
