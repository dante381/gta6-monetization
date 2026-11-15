"""Post-stream run report: clips made, scores, what was published.

Stats are computed locally from state.db and the event log; one optional claude -p call (low
effort, no tools) turns them into a short readable summary. Claude failure falls back to a
plain-text table. Output: data/reports/<stream_id>.md
"""
from __future__ import annotations

import json
import logging
from collections import Counter

from . import claude_cli, config as config_mod, paths
from .state import StateDB

log = logging.getLogger("report")


def gather(state: StateDB, stream_id: str) -> dict:
    cands = state.candidates(stream_id)
    chunks = state.chunks(stream_id)
    queue = state.queue_items()
    routes = Counter(c.route for c in cands)
    tracks = Counter(c.track for c in cands if c.route != "drop")
    top = sorted(cands, key=lambda c: -c.final_score)[:10]
    return {
        "stream_id": stream_id,
        "chunks": len(chunks),
        "candidates": len(cands),
        "routes": dict(routes),
        "tracks_selected": dict(tracks),
        "queue": dict(Counter(q.status for q in queue)),
        "top": [{"id": c.id, "score": round(c.final_score, 1), "track": c.track, "route": c.route,
                 "title": c.meta.title, "events": c.events[:4]} for c in top],
    }


def plain(stats: dict) -> str:
    lines = [f"# Run report: {stats['stream_id']}", "",
             f"- chunks recorded: {stats['chunks']}",
             f"- candidates: {stats['candidates']} (routes: {stats['routes']})",
             f"- selected by track: {stats['tracks_selected']}",
             f"- publish queue: {stats['queue']}", "", "## Top clips", ""]
    for t in stats["top"]:
        lines.append(f"- {t['score']:5.1f}  [{t['route']}/{t['track']}]  {t['title'] or t['id']}")
    return "\n".join(lines) + "\n"


def make_report(state: StateDB, cfg, stream_id: str | None = None, use_claude: bool = True) -> str:
    stream_id = stream_id or state.current_stream() or "unknown"
    stats = gather(state, stream_id)
    text = plain(stats)
    if use_claude:
        try:
            prompt = ("Write a short run report (max 15 lines, markdown) for a GTA 6 stream auto-clipper run. "
                      "Mention clips made, score distribution, what was queued/published, and anything odd. "
                      "Data (JSON):\n" + json.dumps(stats))
            out = claude_cli.call(prompt, schema=None, effort=cfg.claude.effort_report, model=cfg.claude.model,
                                  tools="", timeout_s=cfg.claude.timeout_s)
            if isinstance(out, str) and out.strip():
                text = out.strip() + "\n\n---\n\n" + text
        except claude_cli.ClaudeError as e:
            log.warning("report: claude unavailable, using plain report (%s)", e)
    paths.DATA.reports.mkdir(parents=True, exist_ok=True)
    p = paths.DATA.reports / f"{stream_id}.md"
    p.write_text(text, encoding="utf-8")
    return str(p)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="clipper report")
    ap.add_argument("--stream")
    ap.add_argument("--no-claude", action="store_true")
    a = ap.parse_args(argv)
    cfg = config_mod.load_config()
    print(make_report(StateDB(), cfg, a.stream, use_claude=not a.no_claude))
    return 0
