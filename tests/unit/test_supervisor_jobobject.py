"""Job object: hard memory cap for a process tree, suspended spawn, kill on close."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time

import psutil
import pytest

from clipper.supervisor import jobobject as jo
from clipper.supervisor.jobobject import JobObject, spawn_in_job

ALLOC = ("import sys\n"
         "try:\n"
         "    b = bytearray({mb} * 1024 * 1024)\n"
         "except MemoryError:\n"
         "    sys.exit(7)\n"
         "sys.exit(0)\n")


def _run(job, code, timeout=60):
    p = spawn_in_job([sys.executable, "-c", code], job, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.wait(timeout=timeout)


def test_struct_layout_64bit():
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(jo.IO_COUNTERS) == 48
    assert ctypes.sizeof(jo.JOBOBJECT_BASIC_LIMIT_INFORMATION) == 64
    assert ctypes.sizeof(jo.JOBOBJECT_EXTENDED_LIMIT_INFORMATION) == 144
    assert jo.JOBOBJECT_EXTENDED_LIMIT_INFORMATION.JobMemoryLimit.offset == 120
    assert jo.JOBOBJECT_EXTENDED_LIMIT_INFORMATION.PeakJobMemoryUsed.offset == 136


def test_limits_are_set():
    with JobObject(200) as job:
        info = job.query_limits()
        flags = info.BasicLimitInformation.LimitFlags
        assert flags & jo.JOB_OBJECT_LIMIT_JOB_MEMORY
        assert flags & jo.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        assert not flags & jo.JOB_OBJECT_LIMIT_BREAKAWAY_OK
        assert info.JobMemoryLimit == 200 * 1024 * 1024


def test_child_over_limit_fails_under_limit_succeeds():
    with JobObject(200) as job:
        assert _run(job, ALLOC.format(mb=400)) != 0
        assert _run(job, ALLOC.format(mb=50)) == 0
        assert job.peak_memory() > 50 * 1024 * 1024


def test_grandchild_is_limited_too():
    parent = ("import subprocess, sys\n"
              f"code = {ALLOC.format(mb='{mb}')!r}\n"
              "r1 = subprocess.run([sys.executable, '-c', code.format(mb=400)]).returncode\n"
              "r2 = subprocess.run([sys.executable, '-c', code.format(mb=50)]).returncode\n"
              "sys.exit(10 * (r1 != 0) + (r2 != 0))\n")
    with JobObject(200) as job:
        rc = _run(job, parent)
        assert rc == 10, rc  # 400 MB grandchild failed, 50 MB grandchild worked
        assert job.peak_memory() > 0


def test_spawned_process_is_in_job_and_resumed():
    with JobObject(500) as job:
        p = spawn_in_job([sys.executable, "-c", "import time; time.sleep(30)"], job,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            assert job.contains(p.pid)
            assert p.pid in job.pids()
            # it actually runs (not left suspended): threads are scheduled and CPU time advances
            st = psutil.Process(p.pid).status()
            assert st != psutil.STATUS_STOPPED
            assert job.current_memory() > 0
        finally:
            p.kill()
            p.wait(10)


def test_kill_on_job_close():
    job = JobObject(500)
    p = spawn_in_job([sys.executable, "-c", "import time; time.sleep(60)"], job,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)
    assert p.poll() is None
    job.close()
    assert p.wait(timeout=10) is not None


def test_adopt_descendants_and_not_in_job():
    with JobObject(500) as job:
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            assert not job.contains(p.pid)
            adopted = job.adopt_descendants()
            assert p.pid in adopted
            assert job.contains(p.pid)
        finally:
            p.kill()
            p.wait(10)


def test_assign_failure_kills_child(monkeypatch):
    with JobObject(500) as job:
        spawned = []
        real_popen = subprocess.Popen

        def popen(*a, **kw):
            pr = real_popen(*a, **kw)
            spawned.append(pr)
            return pr

        monkeypatch.setattr(jo.subprocess, "Popen", popen)
        monkeypatch.setattr(job, "assign", lambda pid: (_ for _ in ()).throw(jo.JobError(5, "denied")))
        with pytest.raises(jo.JobError):
            spawn_in_job([sys.executable, "-c", "print(1)"], job, stdout=subprocess.DEVNULL)
        assert spawned and spawned[0].poll() is not None
