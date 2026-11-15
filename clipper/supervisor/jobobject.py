"""Windows job object (ctypes, no pywin32): one hard memory cap for recorder + worker + every
ffmpeg / claude child they start.

    job = JobObject(2048)
    proc = spawn_in_job([python, "-m", "clipper", "work"], job, cwd=..., env=...)

`spawn_in_job` starts the process suspended, assigns it to the job, then resumes it, so the
process never runs a single instruction outside the job. Children inherit the job (no
breakaway flag is set). The job is created with KILL_ON_JOB_CLOSE, so if the supervisor dies
its handle closes and everything inside the job dies with it (no orphaned ffmpeg).
"""
from __future__ import annotations

import ctypes
import logging
import os
import subprocess
from ctypes import wintypes
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- constants
JobObjectBasicProcessIdList = 3
JobObjectExtendedLimitInformation = 9

JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x00001000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_SUSPEND_RESUME = 0x0800
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
THREAD_SUSPEND_RESUME = 0x0002

CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

SPAWN_FLAGS = CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP


# ---------------------------------------------------------------- structs (64-bit correct)
class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),   # LARGE_INTEGER
        ("PerJobUserTimeLimit", ctypes.c_longlong),       # LARGE_INTEGER
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),                    # ULONG_PTR
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


# ---------------------------------------------------------------- function prototypes
# Private WinDLL instances so our argtypes never clash with other modules using ctypes.windll.
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_ntdll = ctypes.WinDLL("ntdll")

_k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
_k32.CreateJobObjectW.restype = wintypes.HANDLE
_k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
_k32.SetInformationJobObject.restype = wintypes.BOOL
_k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
                                           ctypes.POINTER(wintypes.DWORD)]
_k32.QueryInformationJobObject.restype = wintypes.BOOL
_k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_k32.AssignProcessToJobObject.restype = wintypes.BOOL
_k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
_k32.TerminateJobObject.restype = wintypes.BOOL
_k32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
_k32.IsProcessInJob.restype = wintypes.BOOL
_k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.OpenThread.restype = wintypes.HANDLE
_k32.ResumeThread.argtypes = [wintypes.HANDLE]
_k32.ResumeThread.restype = wintypes.DWORD
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.CloseHandle.restype = wintypes.BOOL
_ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
_ntdll.NtResumeProcess.restype = ctypes.c_long  # NTSTATUS


class JobError(OSError):
    pass


def _err(what: str) -> JobError:
    code = ctypes.get_last_error()
    return JobError(code, f"{what} failed: {ctypes.FormatError(code).strip()} (winerror {code})")


def _open_process(pid: int, access: int) -> int:
    h = _k32.OpenProcess(access, False, int(pid))
    if not h:
        raise _err(f"OpenProcess({pid})")
    return h


# ---------------------------------------------------------------- job object
class JobObject:
    """Anonymous job object with a total committed-memory cap for every process inside it."""

    def __init__(self, memory_limit_mb: int | float | None, *, kill_on_close: bool = True,
                 die_on_unhandled_exception: bool = True):
        h = _k32.CreateJobObjectW(None, None)
        if not h:
            raise _err("CreateJobObjectW")
        self.handle: int | None = h
        self.memory_limit_mb = memory_limit_mb
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = 0
        if memory_limit_mb:
            flags |= JOB_OBJECT_LIMIT_JOB_MEMORY
            info.JobMemoryLimit = int(memory_limit_mb * 1024 * 1024)
        if kill_on_close:
            flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if die_on_unhandled_exception:  # no Windows Error Reporting dialog holding a dead process
            flags |= JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        info.BasicLimitInformation.LimitFlags = flags
        if not _k32.SetInformationJobObject(h, JobObjectExtendedLimitInformation, ctypes.byref(info),
                                            ctypes.sizeof(info)):
            e = _err("SetInformationJobObject")
            _k32.CloseHandle(h)
            self.handle = None
            raise e

    # -------------------------------------------------------------- info
    def _h(self) -> int:
        if not self.handle:
            raise JobError(0, "job object is closed")
        return self.handle

    def query_limits(self) -> JOBOBJECT_EXTENDED_LIMIT_INFORMATION:
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        if not _k32.QueryInformationJobObject(self._h(), JobObjectExtendedLimitInformation, ctypes.byref(info),
                                              ctypes.sizeof(info), None):
            raise _err("QueryInformationJobObject")
        return info

    def peak_memory(self) -> int:
        """Peak committed memory (bytes) of all processes in the job combined, since creation."""
        return int(self.query_limits().PeakJobMemoryUsed)

    def pids(self, max_ids: int = 512) -> list[int]:
        class _List(ctypes.Structure):
            _fields_ = [("NumberOfAssignedProcesses", wintypes.DWORD),
                        ("NumberOfProcessIdsInList", wintypes.DWORD),
                        ("ProcessIdList", ctypes.c_size_t * max_ids)]
        lst = _List()
        if not _k32.QueryInformationJobObject(self._h(), JobObjectBasicProcessIdList, ctypes.byref(lst),
                                              ctypes.sizeof(lst), None):
            raise _err("QueryInformationJobObject(pid list)")
        return [int(lst.ProcessIdList[i]) for i in range(lst.NumberOfProcessIdsInList)]

    def current_memory(self) -> int:
        """Sum of private bytes of the processes currently in the job (best effort, via psutil)."""
        import psutil
        total = 0
        for pid in self.pids():
            try:
                total += int(getattr(psutil.Process(pid).memory_info(), "private", 0))
            except (psutil.Error, OSError):
                pass
        return total

    def contains(self, pid: int) -> bool:
        try:
            h = _open_process(pid, PROCESS_QUERY_LIMITED_INFORMATION)
        except JobError:
            return False
        try:
            res = wintypes.BOOL(False)
            if not _k32.IsProcessInJob(h, self._h(), ctypes.byref(res)):
                return False
            return bool(res.value)
        finally:
            _k32.CloseHandle(h)

    # -------------------------------------------------------------- control
    def assign(self, pid: int | subprocess.Popen) -> None:
        pid = pid.pid if isinstance(pid, subprocess.Popen) else int(pid)
        h = _open_process(pid, PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION)
        try:
            if not _k32.AssignProcessToJobObject(self._h(), h):
                raise _err(f"AssignProcessToJobObject({pid})")
        finally:
            _k32.CloseHandle(h)

    def adopt_descendants(self, root_pid: int | None = None, skip: Iterable[int] = ()) -> list[int]:
        """Assign every live descendant of root_pid (default: this process) that isn't in the job
        yet. Used while the supervisor itself starts claude / git / pytest during a repair."""
        import psutil
        skip = set(skip)
        adopted: list[int] = []
        try:
            kids = psutil.Process(root_pid or os.getpid()).children(recursive=True)
        except psutil.Error:
            return adopted
        for k in kids:
            if k.pid in skip or self.contains(k.pid):
                continue
            try:
                self.assign(k.pid)
                adopted.append(k.pid)
            except JobError:  # exited already, or access denied: nothing to do
                pass
        return adopted

    def terminate(self, exit_code: int = 1) -> None:
        if self.handle:
            _k32.TerminateJobObject(self.handle, exit_code)

    def close(self) -> None:
        """Close the handle. With kill_on_close this kills every process still in the job."""
        if self.handle:
            _k32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> "JobObject":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - safety net
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------- spawning
def resume_process(pid: int) -> None:
    """Resume a process created with CREATE_SUSPENDED (NtResumeProcess, fallback per-thread)."""
    try:
        h = _open_process(pid, PROCESS_SUSPEND_RESUME)
    except JobError:
        h = None
    if h:
        try:
            if _ntdll.NtResumeProcess(h) >= 0:
                return
        finally:
            _k32.CloseHandle(h)
    import psutil
    resumed = 0
    for t in psutil.Process(pid).threads():
        th = _k32.OpenThread(THREAD_SUSPEND_RESUME, False, t.id)
        if not th:
            continue
        try:
            if _k32.ResumeThread(th) != 0xFFFFFFFF:
                resumed += 1
        finally:
            _k32.CloseHandle(th)
    if not resumed:
        raise _err(f"ResumeThread({pid})")


def spawn_in_job(cmd: Sequence[str], job: JobObject, **popen_kw) -> subprocess.Popen:
    """Popen suspended, assign to `job`, resume. Raises (and kills the child) if assignment fails."""
    flags = popen_kw.pop("creationflags", 0) | SPAWN_FLAGS
    flags &= ~CREATE_BREAKAWAY_FROM_JOB
    proc = subprocess.Popen(list(map(str, cmd)), creationflags=flags, **popen_kw)
    try:
        job.assign(proc.pid)
        resume_process(proc.pid)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
        raise
    return proc
