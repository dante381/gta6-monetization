"""Supervisor launch-day mode: keeps going across stream ends, winds down once at the window end."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clipper import runwindow
from clipper.config import load_config
from clipper.supervisor.supervisor import Supervisor
from tests.unit.test_supervisor_supervisor import (events, no_real_awake, plain_spawn,  # noqa: F401
                                                   run_with_timeout, script)


def wcfg(end_in_s: float, grace_min: float = 1.0):
    now = datetime.now(timezone.utc)
    win = {"start": (now - timedelta(hours=1)).isoformat(), "end": (now + timedelta(seconds=end_in_s)).isoformat(),
           "drain_grace_min": grace_min, "uploads": True}
    return load_config(overrides={
        "locked": {"run_window": win},
        "supervisor": {"backoff_s": [0.2], "hang_s": 30, "startup_grace_s": 30, "poll_s": 0.05, "mem_log_s": 5,
                       "window_restart_delay_s": 0.2, "window_stop_kill_s": 0.5},
        "health": {"enabled": False},
    })


@pytest.fixture
def no_task(monkeypatch):
    removed = []
    monkeypatch.setattr(runwindow, "remove", lambda *a, **k: removed.append(1) or True)
    return removed


def make(tmp_path, state, cfg, rec, wrk):
    return Supervisor(cfg, "http://example/live", state, spawn=plain_spawn, repair=False, window=True,
                      commands={"recorder": rec, "worker": wrk})


def test_restarts_across_stream_ends_then_stops_for_good(tmp_path, state, no_task):
    rec = script(tmp_path, "rec", "import time; time.sleep(0.3)")          # a "stream" that ends fast
    wrk = script(tmp_path, "wrk", "import time; time.sleep(0.2)")
    sup = make(tmp_path, state, wcfg(end_in_s=4), rec, wrk)
    run_with_timeout(sup, timeout=60)
    assert sup.recorder.starts >= 2, "recorder must be restarted after each stream end while the window is open"
    assert sup.worker.starts >= 2, "idle worker must be brought back for the next stream"
    assert events("window_end") and events("window_finished")
    assert runwindow.is_consumed()
    assert no_task == [1]                                                  # scheduled task retired
    assert sup.recorder.status in ("done", "stopped") and sup.worker.status in ("done", "stopped")


def test_stop_request_ignored_recorder_is_killed(tmp_path, state, no_task):
    rec = script(tmp_path, "rec", "import time; time.sleep(600)")          # never stops by itself
    wrk = script(tmp_path, "wrk", "import time; time.sleep(0.2)")
    sup = make(tmp_path, state, wcfg(end_in_s=2), rec, wrk)
    run_with_timeout(sup, timeout=60)
    assert state.kv_get("recorder.stop") is True
    assert state.kv_get("recorder.done") is True
    assert runwindow.is_consumed() and no_task == [1]


def test_window_not_consumed_by_a_plain_stop(tmp_path, state, no_task):
    """Ctrl-C / reboot mid-day must NOT burn the window (the logon trigger resumes it)."""
    import threading
    rec = script(tmp_path, "rec", "import time; time.sleep(600)")
    wrk = script(tmp_path, "wrk", "import time; time.sleep(600)")
    sup = make(tmp_path, state, wcfg(end_in_s=3600), rec, wrk)
    t = threading.Thread(target=sup.run, daemon=True)
    t.start()
    import time
    time.sleep(1.0)
    sup.request_stop("test")
    t.join(30)
    assert not t.is_alive()
    assert not runwindow.is_consumed() and no_task == []


def test_grace_expiry_forces_stop(tmp_path, state, no_task):
    """A worker that never drains can't keep the supervisor alive past the grace period."""
    rec = script(tmp_path, "rec", "import time; time.sleep(0.2)")
    wrk = script(tmp_path, "wrk", "import time; time.sleep(600)")          # never finishes
    sup = make(tmp_path, state, wcfg(end_in_s=1, grace_min=0.05), rec, wrk)   # 3 s grace
    run_with_timeout(sup, timeout=60)
    assert events("window_grace_expired")
    assert runwindow.is_consumed()
