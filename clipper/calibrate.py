"""Calibration tooling (phase 5): export the top 20 and bottom 20 produced clips for review.

`python -m clipper calibrate export [--stream ID]` writes data/reports/calibration/<stream>/
  review.csv   clip_id, score, track, route, title, would_post (blank: you fill in y/n)
  review.html  playable list with the same rows
  clips/       copies of the mp4s (hard links when possible)
Give the marked CSV back to Claude Code, which adjusts config/settings.yaml (weights, penalties).
`python -m clipper calibrate summary FILE.csv` prints how marks line up with scores.
"""
from __future__ import annotations

import csv
import html
import os
import shutil
from pathlib import Path

from . import paths
from .state import StateDB


def _clip_file(cand_id: str) -> Path | None:
    for d in (paths.DATA.ready, paths.DATA.backup):
        p = d / f"{cand_id}.mp4"
        if p.exists():
            return p
    return None


def export(state: StateDB, stream_id: str | None = None, n: int = 20) -> Path:
    stream_id = stream_id or state.current_stream() or "unknown"
    cands = [c for c in state.candidates(stream_id) if _clip_file(c.id)]
    cands.sort(key=lambda c: -c.final_score)
    picked = cands[:n] + [c for c in cands[n:][-n:]]
    out = paths.DATA.reports / "calibration" / stream_id
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rows = []
    for c in picked:
        src = _clip_file(c.id)
        dst = out / "clips" / src.name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        rows.append({"clip_id": c.id, "score": round(c.final_score, 1), "signal": round(c.signal_score, 1),
                     "track": c.track, "route": c.route, "title": c.meta.title, "scene": c.scene_tag or "",
                     "events": "; ".join(c.events[:5]), "would_post": ""})
    with open(out / "review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["clip_id"])
        w.writeheader()
        w.writerows(rows)
    cells = "".join(
        f"<tr><td>{r['score']}</td><td>{r['track']}</td><td>{html.escape(r['title'])}</td>"
        f"<td><video src='clips/{r['clip_id']}.mp4' controls preload='none' height='320'></video></td></tr>"
        for r in rows)
    (out / "review.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Calibration review</title>"
        "<style>body{font:14px sans-serif}td{padding:6px;vertical-align:top}</style>"
        f"<h1>{html.escape(stream_id)}: top {n} and bottom {n}</h1>"
        "<p>Mark <b>would_post</b> (y/n) in review.csv.</p>"
        f"<table border=1 cellspacing=0><tr><th>score</th><th>track</th><th>title</th><th>clip</th></tr>{cells}</table>",
        encoding="utf-8")
    return out


def summarize(csv_path: str | Path) -> dict:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    marked = [r for r in rows if r.get("would_post", "").strip().lower() in ("y", "n", "yes", "no")]
    yes = [float(r["score"]) for r in marked if r["would_post"].strip().lower().startswith("y")]
    no = [float(r["score"]) for r in marked if r["would_post"].strip().lower().startswith("n")]
    return {"marked": len(marked), "yes": len(yes), "no": len(no),
            "yes_min": min(yes) if yes else None, "no_max": max(no) if no else None,
            "suggested_threshold": round((min(yes) + max(no)) / 2, 1) if yes and no and min(yes) > max(no) else None}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="clipper calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--stream")
    e.add_argument("-n", type=int, default=20)
    s = sub.add_parser("summary")
    s.add_argument("csv")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(export(StateDB(), a.stream, a.n))
    else:
        print(summarize(a.csv))
    return 0
