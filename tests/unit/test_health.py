"""Health monitor: checks, de-dup, watch stdout contract, status file, notify rate limit."""
from __future__ import annotations

import json
import time

import pytest

from clipper import health, heartbeat, notify, paths
from clipper.config import load_config
from clipper.health import Finding
from clipper.models import Chunk, QueueItem


@pytest.fixture
def hcfg_():
    return load_config()


def ts(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + ",123"


def write_log(name, lines):
    """lines: (age_s, LEVEL, message) -> data/logs/<name>"""
    now = time.time()
    with open(paths.DATA.logs / name, "a", encoding="utf-8") as f:
        for age, lvl, msg in lines:
            f.write(f"{ts(now - age)} {lvl} clipper.x: {msg}\n")


def write_events(evs):
    now = time.time()
    with open(paths.DATA.logs / "clips.jsonl", "a", encoding="utf-8") as f:
        for age, kind, fields in evs:
            f.write(json.dumps({"ts": now - age, "kind": kind, **fields}) + "\n")


def only(cfg, state, code):
    return [f for f in health.check_all(cfg, state) if f.code == code]


# ------------------------------------------------------------------ individual checks
def test_clean_state_has_no_findings(hcfg_, state):
    fs = health.check_all(hcfg_, state)
    assert [f for f in fs if f.level != "info"] == []


def test_stuck_stage_fires_and_not(hcfg_, state):
    heartbeat.beat("worker", stage="produce", cand="c1")
    assert only(hcfg_, state, "stuck_stage") == []          # first sighting records the signature
    sig = state.kv_get("health.stage_sig")
    assert sig["sig"][0] == "produce"
    state.kv_set("health.stage_sig", {"sig": sig["sig"], "since": time.time() - 100})
    assert only(hcfg_, state, "stuck_stage") == []          # unchanged but not long enough
    state.kv_set("health.stage_sig", {"sig": sig["sig"], "since": time.time() - 1000})
    f = only(hcfg_, state, "stuck_stage")
    assert len(f) == 1 and f[0].level == "critical" and f[0].proc == "worker" and "produce" in f[0].message
    heartbeat.beat("worker", stage="produce", cand="c2")    # progress: signature changes
    assert only(hcfg_, state, "stuck_stage") == []


def test_stuck_stage_ignores_idle_and_dead_heartbeat(hcfg_, state):
    heartbeat.beat("worker", stage="loop")
    pid = heartbeat.read("worker")["pid"]
    state.kv_set("health.stage_sig", {"sig": ["loop", None, None, pid], "since": time.time() - 5000})
    assert only(hcfg_, state, "stuck_stage") == []
    heartbeat.beat("worker", stage="render")
    p = paths.DATA.heartbeat / "worker.json"
    d = json.loads(p.read_text())
    d["ts"] -= 3600                                          # stale heartbeat: process_down's job
    p.write_text(json.dumps(d))
    state.kv_set("health.stage_sig", {"sig": ["render", None, None, d["pid"]], "since": time.time() - 5000})
    assert only(hcfg_, state, "stuck_stage") == []


def test_gpu_wait_loop(hcfg_, state):
    write_log("worker.log", [(300, "WARNING", "GPU unavailable (attempt 1), waiting 30s: x")] * 5)
    assert only(hcfg_, state, "gpu_wait_loop") == []         # old lines
    write_log("worker.log", [(i * 10, "WARNING", "GPU unavailable (attempt 9), waiting 30s: x") for i in range(2)])
    assert only(hcfg_, state, "gpu_wait_loop") == []         # only 2
    write_log("worker.log", [(5, "WARNING", "GPU unavailable (attempt 10), waiting 30s: x")])
    f = only(hcfg_, state, "gpu_wait_loop")
    assert len(f) == 1 and f[0].proc == "worker" and f[0].key == "gpu_wait_loop:worker"


def test_error_burst(hcfg_, state):
    write_log("worker.log", [(10, "ERROR", f"boom {i}") for i in range(4)])
    assert only(hcfg_, state, "error_burst") == []
    write_log("worker.log", [(10, "ERROR", "boom 99")])
    f = only(hcfg_, state, "error_burst")
    assert len(f) == 1 and "boom N" in f[0].message and "5 errors" in f[0].message


def test_error_burst_counts_tracebacks_once_and_ignores_old(hcfg_, state):
    with open(paths.DATA.logs / "recorder.log", "w", encoding="utf-8") as f:
        for i in range(4):  # ERROR + its Traceback line = one error each
            f.write(f"{ts(time.time() - 5)} ERROR clipper.r: failed {i}\nTraceback (most recent call last):\n  x\n")
    assert only(hcfg_, state, "error_burst") == []
    with open(paths.DATA.logs / "recorder.log", "a", encoding="utf-8") as f:
        f.write(f"{ts(time.time() - 3)} INFO clipper.r: handled\nTraceback (most recent call last):\n")
    assert len(only(hcfg_, state, "error_burst")) == 1       # 4 + a bare traceback = 5
    (paths.DATA.logs / "recorder.log").unlink()
    write_log("recorder.log", [(4000, "ERROR", "old")] * 9)
    assert only(hcfg_, state, "error_burst") == []


def test_repeated_failure(hcfg_, state):
    write_events([(10, "clip_failed", {"error": "FfmpegError: bad"})] * 2)
    assert only(hcfg_, state, "repeated_failure") == []
    write_events([(20, "clip_failed", {"error": "FfmpegError: bad"}), (30, "clip_failed", {"error": "other"}),
                  (9000, "clip_failed", {"error": "FfmpegError: bad"})])
    f = only(hcfg_, state, "repeated_failure")
    assert len(f) == 1 and "x3" in f[0].message


def test_no_new_chunks(hcfg_, state):
    heartbeat.beat("recorder", phase="recording")
    assert only(hcfg_, state, "no_new_chunks") == []
    pid = heartbeat.read("recorder")["pid"]
    state.kv_set("health.recording_since", {"pid": pid, "since": time.time() - 2000})
    f = only(hcfg_, state, "no_new_chunks")
    assert len(f) == 1 and f[0].proc == "recorder"
    state.add_chunk(Chunk(idx=0, path="x.ts", duration_s=5, stream_id="test-stream", recorded_at=time.time() - 10))
    assert only(hcfg_, state, "no_new_chunks") == []
    state.kv_set("health.recording_since", {"pid": pid, "since": time.time() - 2000})
    heartbeat.beat("recorder", phase="waiting")
    assert only(hcfg_, state, "no_new_chunks") == []


def test_backlog(hcfg_, state):
    for i in range(4):
        state.add_chunk(Chunk(idx=i, path="x", duration_s=1, stream_id="test-stream", recorded_at=time.time()))
    assert only(hcfg_, state, "backlog") == []
    state.add_chunk(Chunk(idx=4, path="x", duration_s=1, stream_id="test-stream", recorded_at=time.time()))
    assert len(only(hcfg_, state, "backlog")) == 1


def test_backlog_by_age(hcfg_, state):
    state.add_chunk(Chunk(idx=0, path="x", duration_s=1, stream_id="test-stream", recorded_at=time.time() - 2000))
    assert len(only(hcfg_, state, "backlog")) == 1
    state.update_chunk("test-stream", 0, status="processed")
    assert only(hcfg_, state, "backlog") == []


def test_process_down(hcfg_, state):
    heartbeat.beat("supervisor", procs={"worker": {"status": "running", "uptime_s": 9999},
                                        "recorder": {"status": "backoff", "uptime_s": 0}})
    f = only(hcfg_, state, "process_down")
    assert [x.key for x in f] == ["process_down:worker"]
    heartbeat.beat("worker", stage="x")
    assert only(hcfg_, state, "process_down") == []
    heartbeat.clear("supervisor")                            # no supervisor: skip
    heartbeat.clear("worker")
    assert only(hcfg_, state, "process_down") == []


def test_process_down_respects_startup_grace(hcfg_, state):
    heartbeat.beat("supervisor", procs={"worker": {"status": "running", "uptime_s": 10}})
    assert only(hcfg_, state, "process_down") == []


def test_disk_low(hcfg_, state, monkeypatch):
    import shutil
    from collections import namedtuple
    U = namedtuple("U", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(10**12, 0, 5 * 2**30))
    assert len(only(hcfg_, state, "disk_low")) == 1
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(10**12, 0, 500 * 2**30))
    assert only(hcfg_, state, "disk_low") == []


def test_memory_near_cap(hcfg_, state):
    write_events([(5, "job_memory", {"peak_mb": 1500, "current_mb": 10, "limit_mb": 2000})])
    assert only(hcfg_, state, "memory_near_cap") == []
    write_events([(1, "job_memory", {"peak_mb": 1900, "current_mb": 10, "limit_mb": 2000})])
    assert len(only(hcfg_, state, "memory_near_cap")) == 1


def test_claude_trouble(hcfg_, state):
    write_events([(10, "claude_unavailable", {})] * 2)
    assert only(hcfg_, state, "claude_trouble") == []
    write_events([(20, "claude_unavailable", {})])
    assert [f.key for f in only(hcfg_, state, "claude_trouble")] == ["claude_trouble:unavailable"]
    write_log("worker.log", [(30, "INFO", "claude call ok in 45.0s (effort=low)")])
    assert len(only(hcfg_, state, "claude_trouble")) == 1
    write_log("worker.log", [(10, "INFO", "claude call ok in 200.5s (effort=low)")])
    assert len(only(hcfg_, state, "claude_trouble")) == 2


def test_no_selection_streak(hcfg_, state):
    write_events([(i, "chunk_processed", {"selected": 0}) for i in range(7, 0, -1)])
    assert only(hcfg_, state, "no_selection_streak") == []
    write_events([(0.5, "chunk_processed", {"selected": 0})])
    f = only(hcfg_, state, "no_selection_streak")
    assert len(f) == 1 and f[0].level == "info"
    write_events([(0.1, "chunk_processed", {"selected": 2})])
    assert only(hcfg_, state, "no_selection_streak") == []


def test_publish_state(hcfg_, state):
    assert only(hcfg_, state, "publish_state") == []
    state.set_publish_hold(True, "repair running")
    state.queue_add(QueueItem(clip_id="a", clip_path="a.mp4", sidecar_path="a.json", final_score=1.0, track="gameplay", kind="short", status="needs_review"))
    state.queue_add(QueueItem(clip_id="b", clip_path="b.mp4", sidecar_path="b.json", final_score=1.0, track="gameplay", kind="short", status="failed"))
    fs = only(hcfg_, state, "publish_state")
    assert {f.key for f in fs} == {"publish_hold", "queue:needs_review", "queue:failed"}
    assert any("repair running" in f.message for f in fs)


def test_pause_and_attention(hcfg_, state):
    from clipper import log
    log.attention("something needs you")
    log.attention("HEALTH WARN x: ignored (our own)")
    fs = health.check_all(hcfg_, state)
    att = [f for f in fs if f.code == "attention"]
    assert len(att) == 1 and "something needs you" in att[0].message
    state.kv_set("health.attn_seen", time.time() + 5)
    assert not [f for f in health.check_all(hcfg_, state) if f.code == "attention"]
    pause = paths.ROOT / hcfg_.locked.kill_switch_file
    assert not pause.exists()
    pause.write_text("")
    try:
        assert any(f.key == "paused" for f in health.check_all(hcfg_, state))
    finally:
        pause.unlink()


# ------------------------------------------------------------------ run_once / dedup / robustness
def _fake_check(findings):
    return lambda ctx: list(findings)


def test_dedup_realert_and_resolved(hcfg_, state, monkeypatch):
    f = Finding("warn", "t", "msg", "t:1")
    cur = [f]
    monkeypatch.setattr(health, "CHECKS", [lambda ctx: list(cur)])
    got = []
    assert health.run_once(hcfg_, state, got.append) == [f]
    assert health.run_once(hcfg_, state, got.append) == []           # de-duplicated
    assert got == [f]
    alerts = state.kv_get("health.alerts")                            # re-alert after realert_s
    alerts["t:1"]["last"] -= 4000
    state.kv_set("health.alerts", alerts)
    assert health.run_once(hcfg_, state, got.append) == [f]
    cur.clear()
    health.run_once(hcfg_, state, got.append)
    evs = [json.loads(line) for line in (paths.DATA.logs / "clips.jsonl").read_text().splitlines()]
    h = [e for e in evs if e["kind"] == "health"]
    assert [e["state"] for e in h] == ["new", "new", "resolved"]
    assert "HEALTH WARN t: msg" in (paths.DATA.logs / "ATTENTION.txt").read_text()
    assert state.kv_get("health.alerts") == {}


def test_info_findings_do_not_raise_attention(hcfg_, state, monkeypatch):
    monkeypatch.setattr(health, "CHECKS", [_fake_check([Finding("info", "i", "m", "i:1")])])
    got = []
    health.run_once(hcfg_, state, got.append)
    assert not (paths.DATA.logs / "ATTENTION.txt").exists()
    assert len(got) == 1  # the notifier is still called; it decides (info => log only)


def test_broken_check_does_not_crash(hcfg_, state, monkeypatch):
    def boom(ctx):
        raise RuntimeError("x")
    monkeypatch.setattr(health, "CHECKS", [boom, _fake_check([Finding("warn", "ok", "m", "ok")])])
    assert [f.code for f in health.run_once(hcfg_, state, lambda f: None)] == ["ok"]


def test_broken_notifier_does_not_crash(hcfg_, state, monkeypatch):
    monkeypatch.setattr(health, "CHECKS", [_fake_check([Finding("warn", "ok", "m", "ok")])])

    def bad(f):
        raise RuntimeError("no")
    assert len(health.run_once(hcfg_, state, bad)) == 1


def test_write_status_atomic(hcfg_, monkeypatch):
    fs = [Finding("warn", "a", "m", "a"), Finding("critical", "b", "m", "b")]
    p = health.write_status(fs)
    d = json.loads(p.read_text())
    assert d["level"] == "critical" and len(d["findings"]) == 2 and "ts" in d
    assert not list(paths.DATA.logs.glob("health*.tmp"))
    assert json.loads(health.write_status([]).read_text())["level"] == "ok"
    import os
    real = os.replace                                         # a failing replace leaves the old file intact
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError()))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    health.write_status(fs)
    monkeypatch.setattr(os, "replace", real)
    assert json.loads(p.read_text())["level"] == "ok"
    assert not list(paths.DATA.logs.glob("health*.tmp"))


# ------------------------------------------------------------------ watch stdout contract
def test_watch_prints_one_line_per_event(hcfg_, state, monkeypatch, capsys):
    cur = [Finding("critical", "stuck_stage", "multi\nline  msg", "k1")]
    monkeypatch.setattr(health, "CHECKS", [lambda ctx: list(cur)])
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 2:
            cur.clear()
    health.watch(hcfg_, state, 7, notifier=lambda f: None, max_iterations=4, sleep=sleep)
    out = capsys.readouterr().out
    assert out.splitlines() == ["HEALTH CRITICAL stuck_stage: multi line msg", "HEALTH RESOLVED stuck_stage"]
    assert sleeps == [7, 7, 7]


def test_watch_survives_broken_pass(hcfg_, state, monkeypatch, capsys):
    monkeypatch.setattr(health, "_cycle", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    health.watch(hcfg_, state, 0, notifier=lambda f: None, max_iterations=2, sleep=lambda s: None)
    assert capsys.readouterr().out == ""


def test_main_exit_codes_and_json(monkeypatch, capsys):
    for fs, rc in (([], 0), ([Finding("info", "a", "m", "a")], 0), ([Finding("warn", "a", "m", "a")], 2),
                   ([Finding("warn", "a", "m", "a"), Finding("critical", "b", "m", "b")], 1)):
        monkeypatch.setattr(health, "check_all", lambda cfg, state, now=None, fs=fs: fs)
        assert health.main([]) == rc
    capsys.readouterr()
    health.main(["--json"])
    assert json.loads(capsys.readouterr().out)["level"] == "critical"


# ------------------------------------------------------------------ notify
def test_notify_rate_limit_and_levels(monkeypatch):
    calls = []
    monkeypatch.setattr(notify, "_run_powershell", lambda t, m, timeout=10: calls.append((t, m)))
    monkeypatch.setattr(notify, "_last_toast", 0.0)
    monkeypatch.setattr(notify.os, "name", "nt")
    notify.notify(Finding("info", "i", "m", "i"), wait=True)
    assert calls == []
    notify.notify(Finding("warn", "w", "m", "w"), min_gap_s=60, wait=True)
    notify.notify(Finding("critical", "c", "m", "c"), min_gap_s=60, wait=True)   # inside the gap
    assert len(calls) == 1 and "w: m" in calls[0][1]
    notify.notify(Finding("critical", "c", "m", "c"), min_gap_s=0, wait=True)
    assert len(calls) == 2


def test_toast_swallows_errors(monkeypatch):
    def bad(*a, **k):
        raise OSError("no powershell")
    monkeypatch.setattr(notify, "_run_powershell", bad)
    monkeypatch.setattr(notify, "_last_toast", 0.0)
    monkeypatch.setattr(notify.os, "name", "nt")
    assert notify.toast("t", "m", wait=True) is True


# ------------------------------------------------------------------ chat checks
def test_chat_down_by_status_and_heartbeat(hcfg_, state):
    assert only(hcfg_, state, "chat_down") == []                       # no supervisor
    heartbeat.beat("supervisor", procs={"chat": {"status": "backoff", "uptime_s": 0}})
    f = only(hcfg_, state, "chat_down")
    assert [x.key for x in f] == ["chat_down"] and f[0].level == "warn"
    heartbeat.beat("supervisor", procs={"chat": {"status": "running", "uptime_s": 9999}})
    assert len(only(hcfg_, state, "chat_down")) == 1                   # running but no heartbeat
    heartbeat.beat("chat", phase="collecting")
    assert only(hcfg_, state, "chat_down") == []
    heartbeat.beat("supervisor", procs={"chat": {"status": "running", "uptime_s": 5}})
    heartbeat.clear("chat")
    assert only(hcfg_, state, "chat_down") == []                       # startup grace
    heartbeat.beat("supervisor", procs={"chat": {"status": "done", "uptime_s": 0}})
    assert only(hcfg_, state, "chat_down") == []                       # exited 0 on purpose
    heartbeat.beat("supervisor", procs={"chat": {"status": "running", "uptime_s": 9999}})
    assert only(load_config(overrides={"chat": {"enabled": False}}), state, "chat_down") == []
    assert only(hcfg_, state, "process_down") == []                    # chat is not "critical"


def test_chat_silent(hcfg_, state):
    state.set_current_stream("s1")
    now = time.time()
    state.add_chunk(Chunk(idx=0, path="x.ts", start_s=0, duration_s=300, recorded_at=now - 3000, stream_id="s1"))
    assert only(hcfg_, state, "chat_silent") == []                     # recorder not recording
    heartbeat.beat("recorder", phase="recording")
    f = only(hcfg_, state, "chat_silent")
    assert len(f) == 1 and f[0].level == "info" and f[0].key == "chat_silent"
    state.chat_add("s1", int(now), 3, 2, 1.0, 0, [])
    assert only(hcfg_, state, "chat_silent") == []
    state.set_current_stream("s2")
    state.add_chunk(Chunk(idx=0, path="y.ts", start_s=0, duration_s=300, recorded_at=now - 400, stream_id="s2"))
    assert only(hcfg_, state, "chat_silent") == []                     # live for < 30 min
