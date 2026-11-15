"""Fake `claude` CLI for tests. Use with CLAUDE_BIN="<venv python> tests/fakes/fake_claude.py"
(claude_cli.claude_bin() runs .py paths with the current interpreter).

Mimics `claude -p --output-format json` envelopes:
  {"type":"result","subtype":"success","is_error":false,"result":"...","structured_output":{...}}

Which answer it gives depends on the --json-schema passed:
  * schema has "frames"  -> grid labels, one entry per timestamp
  * schema has "rating"  -> judge result
  * no schema            -> plain text (report) or repair-agent behaviour (if Edit tool allowed)

Environment knobs:
  FAKE_CLAUDE_MODE      ok (default) | limit | garbage | slow | crash
  FAKE_CLAUDE_SCENARIO  path to JSON: {"grid": {"default": {...}, "ranges": [{"from": s, "to": s, ...fields}]},
                                       "judge": {...fields}, "report": "text"}
                        grid ranges use the tile timestamps (chunk-relative seconds)
  FAKE_CLAUDE_LOG       append one JSON line per call: {"argv": [...], "prompt": "..."}
  FAKE_CLAUDE_REPAIR_SCRIPT  python file executed (cwd = claude cwd) when acting as repair agent

Timestamps for grid answers come from a prompt line "TIMESTAMPS: 0,3,6,..." (contract for
the grid prompt); if absent, every "t=<n>" in the prompt is used.
"""
from __future__ import annotations

import json
import os
import re
import runpy
import sys
import time


def _arg(argv: list[str], name: str) -> str | None:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def _emit(payload: dict, rc: int = 0) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    sys.exit(rc)


def _envelope(result: str = "", structured=None, is_error: bool = False) -> dict:
    env = {"type": "result", "subtype": "error" if is_error else "success", "is_error": is_error,
           "result": result, "total_cost_usd": 0.0, "duration_ms": 5, "num_turns": 1, "session_id": "fake"}
    if structured is not None:
        env["structured_output"] = structured
    return env


def _timestamps(prompt: str) -> list[float]:
    m = re.search(r"TIMESTAMPS:\s*([0-9.,\s]+)", prompt)
    if m:
        return [float(x) for x in re.split(r"[,\s]+", m.group(1).strip()) if x]
    return sorted({float(x) for x in re.findall(r"t=(\d+(?:\.\d+)?)", prompt)})


def _grid(prompt: str, scenario: dict) -> dict:
    g = scenario.get("grid", {})
    default = {"kind": "gameplay", "hud_events": [], "wanted_stars": 0, "title_text": None,
               "action": 4, "scene_tag": "city street"}
    default.update(g.get("default", {}))
    frames = []
    for t in _timestamps(prompt):
        f = dict(default, t=t)
        for r in g.get("ranges", []):
            if r["from"] <= t < r["to"]:
                f.update({k: v for k, v in r.items() if k not in ("from", "to")})
        frames.append(f)
    return {"frames": frames}


def _judge(scenario: dict) -> dict:
    j = {"rating": 8, "title": "Insane police chase in GTA 6", "description": "Wild moment from the stream.",
         "hashtags": ["#GTA6", "#Shorts", "#gaming"], "hook": "5 stars in 30 seconds", "spoiler": False,
         "reason": "fast action"}
    j.update(scenario.get("judge", {}))
    return j


def main() -> None:
    argv = sys.argv[1:]
    prompt = sys.stdin.read()
    log = os.environ.get("FAKE_CLAUDE_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"argv": argv, "prompt": prompt, "cwd": os.getcwd()}) + "\n")
    mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
    if mode == "limit":
        sys.stdout.write(json.dumps(_envelope("Claude usage limit reached. Your limit will reset at 5pm.",
                                              is_error=True)))
        sys.exit(1)
    if mode == "crash":
        sys.stderr.write("fatal: something broke\n")
        sys.exit(2)
    if mode == "slow":
        time.sleep(float(os.environ.get("FAKE_CLAUDE_SLEEP", "30")))
    scenario = {}
    sp = os.environ.get("FAKE_CLAUDE_SCENARIO")
    if sp and os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            scenario = json.load(f)

    schema_s = _arg(argv, "--json-schema")
    if mode == "garbage":
        _emit(_envelope("sorry, I cannot produce JSON today"))
    if schema_s:
        schema = json.loads(schema_s)
        props = schema.get("properties", {})
        if "frames" in props:
            data = _grid(prompt, scenario)
        elif "rating" in props:
            data = _judge(scenario)
        else:
            data = scenario.get("generic", {})
        _emit(_envelope(json.dumps(data), structured=data))

    tools = " ".join(filter(None, [_arg(argv, "--tools"), _arg(argv, "--allowedTools")]))
    if "Edit" in tools:  # behave like the repair agent
        script = os.environ.get("FAKE_CLAUDE_REPAIR_SCRIPT")
        if script:
            runpy.run_path(script, run_name="__main__")
        _emit(_envelope("Applied a fix (fake repair agent)."))
    _emit(_envelope(scenario.get("report", "# Run report\n\nFake report.")))


if __name__ == "__main__":
    main()
