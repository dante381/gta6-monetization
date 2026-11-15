"""Launch-day run window: status, upload gate, consumed flag, Task Scheduler XML."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clipper import runwindow
from clipper.config import load_config
from clipper.publish import publisher

IST = timezone(timedelta(hours=5, minutes=30))
START = datetime(2026, 11, 19, 0, 0, tzinfo=IST)
END = datetime(2026, 11, 20, 0, 0, tzinfo=IST)


def wcfg(**over):
    win = {"start": START.isoformat(), "end": END.isoformat(), "drain_grace_min": 180, "uploads": True}
    win.update(over)
    return load_config(overrides={"locked": {"run_window": win}})


def test_real_locked_config_has_the_launch_window():
    w = runwindow.get_window(load_config())
    assert w["start"] == START and w["end"] == END and w["uploads"] is True and w["drain_grace_min"] == 180


@pytest.mark.parametrize("now,expected", [
    (START - timedelta(seconds=1), "before"),
    (START, "active"),
    (END - timedelta(seconds=1), "active"),
    (END, "draining"),
    (END + timedelta(minutes=179), "draining"),
    (END + timedelta(minutes=180), "over"),
])
def test_status_boundaries(now, expected):
    assert runwindow.status(wcfg(), now) == expected


def test_no_window_means_none_and_no_uploads():
    cfg = load_config(overrides={"locked": {"run_window": None}})
    assert runwindow.status(cfg) == "none" and runwindow.get_window(cfg) is None
    assert runwindow.uploads_allowed(cfg) is False


def test_naive_times_are_the_laptops_local_time():
    d = runwindow._parse("2026-11-19T00:00:00")
    assert d.tzinfo is not None and d.hour == 0


def test_end_before_start_is_rejected():
    with pytest.raises(runwindow.WindowError):
        runwindow.get_window(wcfg(end=START.isoformat()))


# ---------------------------------------------------------------- upload gate
def test_uploads_need_the_window_marker(monkeypatch):
    inside = START + timedelta(hours=5)
    monkeypatch.delenv(runwindow.WINDOW_ENV, raising=False)
    assert runwindow.uploads_allowed(wcfg(), inside) is False           # no marker: never
    monkeypatch.setenv(runwindow.WINDOW_ENV, "1")
    assert runwindow.uploads_allowed(wcfg(), inside) is True
    assert runwindow.uploads_allowed(wcfg(), START - timedelta(minutes=1)) is False   # before
    assert runwindow.uploads_allowed(wcfg(), END + timedelta(minutes=170)) is True    # draining
    assert runwindow.uploads_allowed(wcfg(), END + timedelta(minutes=181)) is False   # over
    assert runwindow.uploads_allowed(wcfg(uploads=False), inside) is False            # window without uploads


def test_publisher_stays_in_test_mode_without_the_marker(monkeypatch):
    now = datetime.now(timezone.utc)
    cfg = wcfg(start=(now - timedelta(hours=1)).isoformat(), end=(now + timedelta(hours=1)).isoformat())
    monkeypatch.delenv(runwindow.WINDOW_ENV, raising=False)
    assert publisher.live_test_mode(cfg) is True                  # a manual run/test inside the window


def test_publisher_allows_uploads_only_in_a_window_run(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setenv(runwindow.WINDOW_ENV, "1")
    inside = wcfg(start=(now - timedelta(hours=1)).isoformat(), end=(now + timedelta(hours=1)).isoformat())
    assert publisher.live_test_mode(inside) is False
    before = wcfg(start=(now + timedelta(hours=1)).isoformat(), end=(now + timedelta(hours=25)).isoformat())
    assert publisher.live_test_mode(before) is True
    after = wcfg(start=(now - timedelta(hours=30)).isoformat(), end=(now - timedelta(hours=6)).isoformat())
    assert publisher.live_test_mode(after) is True


def test_window_cannot_override_a_cfg_that_forces_nothing_else(monkeypatch):
    """Garbage in the window config must fail safe (stay in test mode), not crash."""
    monkeypatch.setenv(runwindow.WINDOW_ENV, "1")
    cfg = wcfg(start="not a date")
    assert publisher.live_test_mode(cfg) is True


# ---------------------------------------------------------------- consumed flag
def test_consumed_lifecycle(isolated_data):
    cfg = wcfg()
    inside = START + timedelta(hours=1)
    assert runwindow.may_start(cfg, inside) == (True, "active")
    runwindow.mark_started()
    assert runwindow.read_state()["runs"] == 1
    runwindow.mark_consumed("test")
    ok, why = runwindow.may_start(cfg, inside)
    assert not ok and "rearm" in why and runwindow.is_consumed()
    runwindow.rearm()
    assert runwindow.may_start(cfg, inside)[0] is True


def test_may_start_refuses_early_late_and_unconfigured():
    cfg = wcfg()
    assert "too early" in runwindow.may_start(cfg, START - timedelta(hours=1))[1]
    assert "ended" in runwindow.may_start(cfg, END + timedelta(hours=4))[1]
    nocfg = load_config(overrides={"locked": {"run_window": None}})
    assert not runwindow.may_start(nocfg)[0]
    assert runwindow.may_start(cfg, END + timedelta(minutes=30)) == (True, "draining")


def test_seconds_until():
    assert runwindow.seconds_until(wcfg(), "start", START - timedelta(hours=2)) == 7200
    assert runwindow.seconds_until(wcfg(), "end", END - timedelta(seconds=30)) == 30


# ---------------------------------------------------------------- Task Scheduler
def test_task_xml_is_a_one_time_waking_task():
    x = runwindow.task_xml(wcfg(), "https://www.youtube.com/channel/UCx/live", user="PC\\me")
    assert "<TimeTrigger>" in x and "<StartBoundary>2026-11-19T00:00:00+05:30</StartBoundary>" in x
    assert "<EndBoundary>2026-11-20T00:00:00+05:30</EndBoundary>" in x
    assert "<WakeToRun>true</WakeToRun>" in x and "<StartWhenAvailable>true</StartWhenAvailable>" in x
    # a reboot during the day restarts it (logon trigger), but only inside the window (EndBoundary);
    # the supervisor's own guard refuses anything outside the window or after it was consumed
    assert "<DeleteExpiredTaskAfter>" in x and x.count("<EndBoundary>2026-11-20T00:00:00+05:30</EndBoundary>") == 2
    assert "<LogonTrigger>" in x and "<RestartOnFailure>" in x
    assert "supervise --window --url https://www.youtube.com/channel/UCx/live" in x
    assert "<ExecutionTimeLimit>PT30H</ExecutionTimeLimit>" in x              # 27 h window + grace, with margin
    assert "pythonw.exe" in x and "<WorkingDirectory>" in x


def test_install_uses_xml_and_cleans_up(tmp_path):
    calls = []

    def runner(cmd):
        calls.append(list(cmd))
        return 0, "SUCCESS", ""

    out = runwindow.install(wcfg(), "https://www.youtube.com/channel/UCx/live", runner=runner)
    assert out["output"] == "SUCCESS"
    assert calls[0][:4] == ["schtasks", "/Create", "/F", "/TN"] and runwindow.TASK_NAME in calls[0]
    assert not (runwindow.paths.TMP / f"{runwindow.TASK_NAME}.task.xml").exists()


def test_install_reports_access_denied_and_refuses_when_consumed(isolated_data):
    def denied(cmd):
        return 1, "", "ERROR: Access is denied."
    with pytest.raises(runwindow.WindowError, match="elevated"):
        runwindow.install(wcfg(), "u", runner=denied)
    runwindow.mark_consumed()
    with pytest.raises(runwindow.WindowError, match="rearm"):
        runwindow.install(wcfg(), "u", runner=lambda c: (0, "", ""))


def test_remove_and_status_parse():
    assert runwindow.remove(runner=lambda c: (0, "", "")) is True
    assert runwindow.remove(runner=lambda c: (1, "", "not found")) is False
    st = runwindow.task_status(runner=lambda c: (0, "TaskName: \\GTA6ClipperLaunchDay\nStatus: Ready\n"
                                                    "Next Run Time: 19-11-2026 00:00:00\n", ""))
    assert st["registered"] and st["Status"] == "Ready" and st["Next Run Time"].startswith("19-11-2026")
    assert runwindow.task_status(runner=lambda c: (1, "", "ERROR"))["registered"] is False
