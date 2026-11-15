"""Supervisor loop: restarts, backoff, hang detection, crash signatures, repair trigger, drain."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time

import pytest

from clipper import crash, heartbeat, paths
from clipper.config import load_config
from clipper.models import Chunk
from clipper.supervisor import nosleep
from clipper.supervisor import supervisor as sv
from clipper.supervisor.supervisor import Supervisor

FAST = {"supervisor": {"backoff_s": [0.3, 0.6, 1.0], "stable_reset_s": 600, "hang_s": 1.5,
                       "startup_grace_s": 0.5, "crash_window_s": 30, "crash_count": 3,
                       "poll_s": 0.05, "mem_log_s": 0.2, "slow_restart_s": 0.5}}


def mk_cfg(**sup):
    o = json.loads(json.dumps(FAST))
    o["supervisor"].update(sup)
    return load_config(overrides=o)


def script(tmp_path, name, body):
    p = tmp_path / f"{name}.py"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(p)]


def plain_spawn(cmd, **kw):
    return subprocess.Popen(cmd, **kw)


def events(kind=None):
    p = paths.DATA.logs / "clips.jsonl"
    if not p.exists():
        return []
    ev = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [e for e in ev if kind is None or e["kind"] == kind]


@pytest.fixture(autouse=True)
def no_real_awake(monkeypatch):
    calls = []
    monkeypatch.setattr(nosleep, "set_state", lambda f: calls.append(f) or 1)
    return calls


def make(tmp_path, state, rec, wrk, cfg=None, agent=None, **kw):
    cfg = cfg or mk_cfg()
    return Supervisor(cfg, "http://example/live", state, spawn=plain_spawn,
                      commands={"recorder": rec, "worker": wrk}, repair_agent=agent, **kw)


class FakeAgent:
    def __init__(self, ok=True, max_attempts=3):
        self.reports, self.ok, self.max, self.used = [], ok, max_attempts, 0

    def exhausted(self):
        return self.used >= self.max

    def attempt(self, report, tail):
        self.reports.append((report, tail))
        if self.exhausted():
            return False
        self.used += 1
        return self.ok


def run_with_timeout(sup, timeout=40):
    import threading
    t = threading.Thread(target=sup.run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        sup.request_stop("test timeout")
        t.join(10)
        pytest.fail("supervisor did not finish in time")


def test_restart_after_crash_with_backoff(tmp_path, state):
    marker = tmp_path / "count.txt"
    wrk = script(tmp_path, "worker", f"""
        import pathlib, sys
        m = pathlib.Path(r"{marker}")
        n = int(m.read_text()) if m.exists() else 0
        m.write_text(str(n + 1))
        sys.exit(3 if n < 2 else 0)
        """)
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    sup = make(tmp_path, state, rec, wrk, once=False, repair=False)
    t0 = time.monotonic()
    run_with_timeout(sup)
    elapsed = time.monotonic() - t0
    assert int(marker.read_text()) == 3
    assert sup.worker.starts == 3
    # backoff 0.3 then 0.6 s between the three runs
    assert elapsed >= 0.9
    delays = [e["delay_s"] for e in events("restart_scheduled") if e["name"] == "worker"]
    assert delays == [0.3, 0.6]
    assert [e["signature"] for e in events("crash")] == ["exit:3", "exit:3"]


def test_recorder_exit_zero_not_restarted_and_worker_drains(tmp_path, state):
    state.add_chunk(Chunk(idx=0, path="x.ts", duration_s=5, stream_id="test-stream"))
    # worker exits 0 only after the chunk is processed (done flag simulated through a file)
    flag = tmp_path / "processed.flag"
    wrk = script(tmp_path, "worker", f"""
        import pathlib, sys, time
        f = pathlib.Path(r"{flag}")
        for _ in range(100):
            if f.exists():
                sys.exit(0)
            time.sleep(0.05)
        sys.exit(9)
        """)
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    sup = make(tmp_path, state, rec, wrk, repair=False)

    import threading
    def finish():
        time.sleep(1.0)
        state.update_chunk("test-stream", 0, status="processed")
        flag.write_text("1")
    threading.Thread(target=finish, daemon=True).start()
    run_with_timeout(sup)
    assert sup.recorder.starts == 1          # never restarted
    assert sup.recorder.status == "done"
    assert sup.worker.starts == 1
    assert events("stream_end")
    assert events("worker_drained")


def test_worker_clean_exit_with_backlog_is_restarted(tmp_path, state):
    state.add_chunk(Chunk(idx=0, path="x.ts", duration_s=5, stream_id="test-stream"))
    marker = tmp_path / "n.txt"
    wrk = script(tmp_path, "worker", f"""
        import pathlib, sys
        m = pathlib.Path(r"{marker}")
        n = int(m.read_text()) if m.exists() else 0
        m.write_text(str(n + 1))
        if n >= 1:
            sys.path.insert(0, r"{paths.ROOT}")
            from clipper.state import StateDB
            db = StateDB(); db.update_chunk("test-stream", 0, status="processed"); db.close()
        sys.exit(0)
        """)
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    sup = make(tmp_path, state, rec, wrk, repair=False)
    run_with_timeout(sup)
    assert int(marker.read_text()) == 2


def test_hang_detected_killed_and_restarted(tmp_path, state):
    marker = tmp_path / "n.txt"
    wrk = script(tmp_path, "worker", f"""
        import os, pathlib, sys, time
        sys.path.insert(0, r"{paths.ROOT}")
        from clipper import heartbeat
        m = pathlib.Path(r"{marker}")
        n = int(m.read_text()) if m.exists() else 0
        m.write_text(str(n + 1))
        if n >= 1:
            sys.exit(0)           # second run is healthy and finishes
        heartbeat.beat("worker", stage="hanging")
        time.sleep(300)           # stops beating: hung
        """)
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    sup = make(tmp_path, state, rec, wrk, repair=False)
    run_with_timeout(sup)
    assert int(marker.read_text()) == 2
    hangs = events("hang")
    assert len(hangs) == 1 and hangs[0]["name"] == "worker"
    crashes = events("crash")
    assert crashes and crashes[0]["signature"] == "hang@hanging"


def test_beating_process_is_not_killed(tmp_path, state):
    wrk = script(tmp_path, "worker", f"""
        import sys, time
        sys.path.insert(0, r"{paths.ROOT}")
        from clipper import heartbeat
        end = time.time() + 3.5
        while time.time() < end:
            heartbeat.beat("worker"); time.sleep(0.2)
        sys.exit(0)
        """)
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    sup = make(tmp_path, state, rec, wrk, repair=False)
    run_with_timeout(sup)
    assert not events("hang")
    assert sup.worker.starts == 1


def test_three_identical_signatures_trigger_repair_then_restart(tmp_path, state):
    marker = tmp_path / "n.txt"
    wrk = script(tmp_path, "worker", f"""
        import pathlib, sys
        sys.path.insert(0, r"{paths.ROOT}")
        from clipper import crash
        m = pathlib.Path(r"{marker}")
        n = int(m.read_text()) if m.exists() else 0
        m.write_text(str(n + 1))
        if n >= 3:
            sys.exit(0)           # fixed after the repair
        crash.install_crash_handler("worker")
        raise ValueError("boom")
        """)
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    state.set_publish_hold(True, "repair")
    state.set_keep_chunks(True)
    agent = FakeAgent(ok=True)
    sup = make(tmp_path, state, rec, wrk, agent=agent)
    run_with_timeout(sup)
    assert len(agent.reports) == 1
    rep, tail = agent.reports[0]
    assert rep["signature"].startswith("ValueError@")
    assert int(marker.read_text()) == 4           # 3 crashes, repair, 4th run exits 0
    # hold released by the supervisor after a successful repair
    assert state.publish_hold() is False
    assert state.keep_chunks() is False
    assert events("repair_applied")


def test_repair_success_keeps_non_repair_hold(tmp_path, state):
    from clipper.supervisor.repair import release_repair_hold
    state.set_publish_hold(True, "auth")
    release_repair_hold(state)
    assert state.publish_hold() is True
    state.set_publish_hold(True, "auth+repair")
    release_repair_hold(state)
    assert state.publish_hold() is True and state.kv_get("hold.publish_reason") == "auth"


def test_recorder_crashes_never_trigger_repair(tmp_path, state):
    marker = tmp_path / "n.txt"
    rec = script(tmp_path, "rec", f"""
        import pathlib, sys
        m = pathlib.Path(r"{marker}")
        n = int(m.read_text()) if m.exists() else 0
        m.write_text(str(n + 1))
        sys.exit(0 if n >= 4 else 5)
        """)
    wrk = script(tmp_path, "worker", "import sys; sys.exit(0)")
    agent = FakeAgent()
    sup = make(tmp_path, state, rec, wrk, cfg=mk_cfg(backoff_s=[0.05]), agent=agent)
    run_with_timeout(sup)
    assert int(marker.read_text()) == 5
    assert agent.reports == []


def test_repair_cap_keeps_recorder_and_slows_worker(tmp_path, state):
    rec_marker = tmp_path / "rec.txt"
    rec = script(tmp_path, "rec", f"""
        import pathlib, sys, time
        sys.path.insert(0, r"{paths.ROOT}")
        from clipper import heartbeat
        m = pathlib.Path(r"{rec_marker}")
        m.write_text("started")
        end = time.time() + 4     # keeps running (and beating) while the worker is capped
        while time.time() < end:
            heartbeat.beat("recorder"); time.sleep(0.2)
        sys.exit(0)
        """)
    wrk = script(tmp_path, "worker", "import sys; sys.exit(4)")
    agent = FakeAgent(ok=False, max_attempts=1)
    cfg = mk_cfg(backoff_s=[0.05], slow_restart_s=0.4)
    sup = make(tmp_path, state, rec, wrk, cfg=cfg, agent=agent)
    run_with_timeout(sup, timeout=60)
    assert sup.worker.capped
    assert len(agent.reports) == 1        # one real attempt (failed); capped worker never re-triggers repair
    assert events("worker_capped")
    assert sup.recorder.starts == 1       # recorder untouched by the worker's trouble
    # stopped once the stream ended; worker was only slow-restarted
    assert sup.worker.starts < 12


def test_once_mode_no_restarts(tmp_path, state):
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    wrk = script(tmp_path, "worker", "import sys; sys.exit(3)")
    sup = make(tmp_path, state, rec, wrk, once=True)
    run_with_timeout(sup)
    assert sup.worker.starts == 1 and sup.recorder.starts == 1


def test_crash_signature_from_report_and_fallback(tmp_path, state):
    wrk = script(tmp_path, "worker", f"""
        import sys
        sys.path.insert(0, r"{paths.ROOT}")
        from clipper import crash
        crash.install_crash_handler("worker")
        raise KeyError("x")
        """)
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    sup = make(tmp_path, state, rec, wrk, once=True)
    run_with_timeout(sup)
    sigs = [e["signature"] for e in events("crash")]
    assert sigs and sigs[0].startswith("KeyError@")


def test_stale_crash_report_ignored(tmp_path, state):
    crash.error_path("worker").parent.mkdir(parents=True, exist_ok=True)
    crash.error_path("worker").write_text(json.dumps({"signature": "Old@x:y", "ts": 1.0}))
    wrk = script(tmp_path, "worker", "import sys; sys.exit(6)")
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    sup = make(tmp_path, state, rec, wrk, once=True)
    run_with_timeout(sup)
    assert events("crash")[0]["signature"] == "exit:6"


def test_keep_awake_and_memory_logging(tmp_path, state, no_real_awake):
    wrk = script(tmp_path, "worker", "import sys, time; time.sleep(1.0); sys.exit(0)")
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    sup = make(tmp_path, state, rec, wrk, repair=False)
    run_with_timeout(sup)
    assert no_real_awake[0] == nosleep.ES_CONTINUOUS | nosleep.ES_SYSTEM_REQUIRED
    assert no_real_awake[-1] == nosleep.ES_CONTINUOUS
    mem = events("job_memory")
    assert mem and mem[-1]["limit_mb"] == 2048
    assert events("supervisor_start") and events("supervisor_stop")


def test_real_job_spawn_default_commands(tmp_path, state, monkeypatch):
    """Default spawn path: processes are created inside the job object."""
    wrk = script(tmp_path, "worker", "import sys; sys.exit(0)")
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    sup = Supervisor(mk_cfg(), "u", state, commands={"recorder": rec, "worker": wrk}, repair=False)
    run_with_timeout(sup)
    assert sup.recorder.starts == 1 and sup.worker.starts == 1
    assert sup.job.handle is None  # closed at shutdown


def test_default_commands():
    c = sv.default_commands("https://y/live")
    assert c["recorder"] == [str(paths.VENV_PY), "-m", "clipper", "record", "--url", "https://y/live"]
    assert c["worker"] == [str(paths.VENV_PY), "-m", "clipper", "work"]


def test_parse_args():
    a = sv.parse_args(["--url", "u", "--no-repair", "--once"])
    assert a.url == "u" and a.no_repair and a.once


def test_health_restarts_stuck_worker_without_counting_a_crash(tmp_path, state):
    marker = tmp_path / "n.txt"
    wrk = script(tmp_path, "worker", f"""
        import pathlib, sys, time
        sys.path.insert(0, r"{paths.ROOT}")
        from clipper import heartbeat
        m = pathlib.Path(r"{marker}")
        n = int(m.read_text()) if m.exists() else 0
        m.write_text(str(n + 1))
        if n >= 1:
            sys.exit(0)           # second run is healthy and finishes
        while True:               # alive and beating, but never progresses
            heartbeat.beat("worker", stage="produce", cand="c1"); time.sleep(0.2)
        """)
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    cfg = load_config(overrides={"supervisor": {**FAST["supervisor"], "hang_s": 5, "startup_grace_s": 5},
                                 "health": {"interval_s": 0.3, "stuck_stage_s": 0.8}})
    seen = []
    sup = make(tmp_path, state, rec, wrk, cfg=cfg, repair=False, notifier=seen.append)
    run_with_timeout(sup)
    assert int(marker.read_text()) == 2
    hr = events("health_restart")
    assert len(hr) == 1 and hr[0]["code"] == "stuck_stage"
    assert not events("crash") and not events("hang")
    assert sup.worker.crash_times == {}
    assert [f.code for f in seen] == ["stuck_stage"]
    assert heartbeat.read("supervisor") is None   # cleared at shutdown


# ------------------------------------------------------------------ chat collector (non-critical third process)
def test_chat_crash_is_non_critical(tmp_path, state):
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    wrk = script(tmp_path, "worker", "import time; time.sleep(1.2)")
    chat = script(tmp_path, "chat", "import sys; sys.exit(3)")
    agent = FakeAgent()
    sup = Supervisor(
        mk_cfg(), "http://example/live", state, spawn=plain_spawn, repair_agent=agent,
        commands={"recorder": rec, "worker": wrk, "chat": chat})
    assert "chat" in sup.procs
    run_with_timeout(sup)
    c = sup.procs["chat"]
    assert c.starts >= 3                      # restarted with backoff after each non-zero exit
    assert agent.reports == []                # never triggers the repair agent
    assert not state.publish_hold()
    assert sup.worker.status == "done"        # stream-end handling was not blocked by chat
    assert c.status == "stopped"              # stopped together with the others


def test_chat_exit_zero_not_restarted_and_disabled_flag(tmp_path, state):
    rec = script(tmp_path, "rec", "import sys; sys.exit(0)")
    wrk = script(tmp_path, "worker", "import time; time.sleep(1.0)")
    chat = script(tmp_path, "chat", "import sys; sys.exit(0)")
    cmds = {"recorder": rec, "worker": wrk, "chat": chat}
    sup = Supervisor(mk_cfg(), "u", state, spawn=plain_spawn, commands=cmds, repair=False)
    run_with_timeout(sup)
    assert sup.procs["chat"].starts == 1 and sup.procs["chat"].status in ("done", "stopped")
    off = load_config(overrides={"chat": {"enabled": False}})
    assert "chat" not in Supervisor(off, "u", state, spawn=plain_spawn, commands=cmds).procs
    assert sv.default_commands("u")["chat"][-2:] == ["clipper", "chat"]
