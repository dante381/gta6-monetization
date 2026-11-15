"""Percentile selection: threshold math, floor, hold / final, idempotence, crash repair, queue row."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from clipper import paths, selection
from clipper.models import Candidate, ClipMeta, ClipSidecar
from clipper.publish import enqueue, publish_tick
from tests.unit.test_publish_tick import NOW, no_real_api, pcfg, yt  # noqa: F401  (fixtures)

SID = "test-stream"


def events(kind):
    p = paths.DATA.logs / "clips.jsonl"
    if not p.exists():
        return []
    return [r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
            if r["kind"] == kind]


def add_backup(state, cfg, cid, score, produce=True, queue=True):
    """A scored candidate + (optionally) its produced backup clip, sidecar and queue row."""
    c = Candidate(id=cid, chunk_idx=0, start_s=0, peak_s=10, end_s=40, track="gameplay", final_score=score,
                  signal_score=score, route="backup", meta=ClipMeta(title="t"))
    state.save_candidate(c, SID, produced=produce)
    state.history_add(cid, SID, score)
    if produce:
        mp4 = paths.DATA.backup / f"{cid}.mp4"
        mp4.write_bytes(b"\x00" * 64)
        sc = ClipSidecar(clip_id=cid, clip_path=str(mp4), track="gameplay", route="backup", final_score=score,
                         signal_score=score, title="t", description="", hashtags=["#Shorts"], hook="", spoiler=False,
                         duration_s=30, start_s=0, end_s=30, created_at=time.time(), candidate=c.to_dict())
        (paths.DATA.backup / f"{cid}.json").write_text(json.dumps(sc.to_dict()), encoding="utf-8")
        if queue:
            enqueue(sc, state, cfg)
    return c


def population(state, cfg, scores, produce=True):
    for i, s in enumerate(scores):
        add_backup(state, cfg, f"c{i:03d}", s, produce=produce)


def ids(res):
    return sorted(c.id for c in res)


def test_percentile_math_and_strict_greater(state, cfg):
    scores = list(range(50, 90))                           # 40 samples, p90 = 85.1
    population(state, cfg, scores)
    thr = float(np.percentile(scores, 90))
    res = selection.decide(cfg, state, SID)
    assert ids(res) == ["c036", "c037", "c038", "c039"]    # scores 86..89 > 85.1
    ev = events("selection")[-1]
    assert ev["threshold"] == pytest.approx(thr, abs=0.01) and ev["n"] == 40 and ev["promoted"] == 4
    assert len(events("promoted")) == 4
    assert {c.id for c in state.candidates(SID, route="publish")} == {c.id for c in res}


def test_ties_never_promote_at_the_threshold(state, cfg):
    population(state, cfg, [70.0] * 40)                    # threshold == 70 == every score: strict > promotes none
    assert selection.decide(cfg, state, SID) == []
    add_backup(state, cfg, "top", 70.5)
    # p90 of 41 samples is still 70 -> only the single higher score is promoted
    assert ids(selection.decide(cfg, state, SID)) == ["top"]


def test_floor_blocks_low_scores(state, cfg):
    scores = [20.0 + i * 0.5 for i in range(40)]           # all below the 50 floor
    population(state, cfg, scores)
    assert max(scores) < cfg.locked.selection.floor
    assert selection.decide(cfg, state, SID) == []
    assert not list(paths.DATA.ready.glob("*.mp4"))


def test_hold_below_min_samples_then_final_decides(state, cfg):
    scores = [60 + i for i in range(12)]                   # 12 samples: < 30, >= 10
    population(state, cfg, scores)
    assert selection.decide(cfg, state, SID) == []
    assert not list(paths.DATA.ready.glob("*.mp4")) and len(list(paths.DATA.backup.glob("*.mp4"))) == 12
    res = selection.decide(cfg, state, SID, final=True)
    thr = float(np.percentile(scores, 90))
    assert ids(res) == [f"c{i:03d}" for i, s in enumerate(scores) if s > thr] and res
    assert events("selection")[-1]["final"] is True


def test_final_with_too_few_samples_promotes_nothing(state, cfg):
    population(state, cfg, [80, 85, 90, 95, 99])           # 5 < end_min_samples
    assert selection.decide(cfg, state, SID, final=True) == []
    ev = events("selection")[-1]
    assert ev["reason"] == "too_few_samples" and ev["n"] == 5
    assert len(list(paths.DATA.backup.glob("*.mp4"))) == 5


def test_promotion_moves_files_rewrites_sidecar_and_queue(state, cfg):
    population(state, cfg, list(range(50, 90)))
    selection.decide(cfg, state, SID)
    cid = "c039"
    assert not (paths.DATA.backup / f"{cid}.mp4").exists() and not (paths.DATA.backup / f"{cid}.json").exists()
    mp4 = paths.DATA.ready / f"{cid}.mp4"
    assert mp4.exists()
    side = json.loads((paths.DATA.ready / f"{cid}.json").read_text(encoding="utf-8"))
    assert side["route"] == "publish" and side["clip_path"] == str(mp4)
    assert side["candidate"]["route"] == "publish"
    q = state.queue_get(cid)
    assert (q.kind, q.status, q.clip_path, q.sidecar_path) == ("publish", "queued", str(mp4),
                                                                 str(paths.DATA.ready / f"{cid}.json"))
    assert state.queue_get("c000").kind == "backup"        # below the cut: stays a backup
    assert not list(paths.DATA.ready.glob("*.tmp"))


def test_idempotent(state, cfg):
    population(state, cfg, list(range(50, 90)))
    first = selection.decide(cfg, state, SID)
    n_ev = len(events("promoted"))
    assert first and selection.decide(cfg, state, SID) == []
    assert len(events("promoted")) == n_ev
    assert len(list(paths.DATA.ready.glob("*.mp4"))) == len(first)


@pytest.mark.parametrize("stage", ["moved_only", "sidecar_written"])
def test_crash_between_file_move_and_db_update_is_repaired(state, cfg, stage):
    population(state, cfg, list(range(50, 90)))
    cid = "c039"
    os.replace(paths.DATA.backup / f"{cid}.mp4", paths.DATA.ready / f"{cid}.mp4")   # crash right after the move
    if stage == "sidecar_written":
        d = json.loads((paths.DATA.backup / f"{cid}.json").read_text(encoding="utf-8"))
        d["route"], d["clip_path"] = "publish", str(paths.DATA.ready / f"{cid}.mp4")
        (paths.DATA.ready / f"{cid}.json").write_text(json.dumps(d), encoding="utf-8")
    assert state.queue_get(cid).kind == "backup"
    selection.decide(cfg, state, SID)
    assert (paths.DATA.ready / f"{cid}.mp4").exists() and not (paths.DATA.backup / f"{cid}.json").exists()
    side = json.loads((paths.DATA.ready / f"{cid}.json").read_text(encoding="utf-8"))
    assert side["route"] == "publish" and side["clip_path"] == str(paths.DATA.ready / f"{cid}.mp4")
    assert state.queue_get(cid).kind == "publish"
    assert cid in {c.id for c in state.candidates(SID, route="publish")}
    assert events("promotion_repaired")


def test_repair_runs_even_when_holding(state, cfg):
    population(state, cfg, [90, 91])                        # far below min_samples: hold
    os.replace(paths.DATA.backup / "c001.mp4", paths.DATA.ready / "c001.mp4")
    assert selection.decide(cfg, state, SID) == []
    assert state.queue_get("c001").kind == "publish"


def test_already_uploaded_is_skipped_and_logged_once(state, cfg):
    population(state, cfg, list(range(50, 90)))
    state.queue_update("c039", status="uploaded")
    res = selection.decide(cfg, state, SID)
    assert ids(res) == ["c036", "c037", "c038"]
    assert (paths.DATA.backup / "c039.mp4").exists()
    assert state.queue_get("c039").kind == "backup"
    selection.decide(cfg, state, SID)
    skipped = events("promote_skipped")
    assert len(skipped) == 1 and skipped[0]["cand"] == "c039"


def test_only_produced_candidates_of_this_stream(state, cfg):
    population(state, cfg, list(range(50, 90)))
    add_backup(state, cfg, "unproduced", 99, produce=False)
    c = Candidate(id="other", chunk_idx=0, start_s=0, peak_s=1, end_s=30, track="gameplay", final_score=99,
                  route="backup", meta=ClipMeta(title="t"))
    state.save_candidate(c, "other-stream", produced=True)
    res = selection.decide(cfg, state, SID)
    assert "unproduced" not in ids(res) and "other" not in ids(res)


def test_backups_never_uploaded_and_promoted_clips_are(state, pcfg, yt):  # noqa: F811
    """Regression: with upload_backups false a kept clip stays out of YouTube until it is promoted."""
    from clipper.config import load_config
    assert load_config().publish.upload_backups is False         # shipped default
    pcfg.publish.upload_backups = False
    population(state, pcfg, list(range(50, 90)))
    assert all(q.kind == "backup" for q in state.queue_items())
    assert publish_tick(pcfg, state, now=NOW, youtube=yt) == [] and yt.inserts == []
    promoted = selection.decide(pcfg, state, SID)
    assert len(promoted) == 4
    res = publish_tick(pcfg, state, now=NOW, youtube=yt)
    assert sorted(r["clip_id"] for r in res) == ids(promoted)
    assert all(c["body"]["status"]["privacyStatus"] == "private" for c in yt.inserts)   # scheduled, not unlisted
    assert state.queue_get("c000").status == "queued" and len(yt.inserts) == 4
