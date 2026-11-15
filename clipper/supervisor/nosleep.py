"""Keep the laptop awake while the supervisor runs (SetThreadExecutionState).

The request is per thread: call keep_awake() and allow_sleep() from the same long-lived thread
(the supervisor's main loop). It doesn't change the power plan; `power_plan_commands()` lists the
one-time powercfg commands from the setup section (never sleep / hibernate on AC) for the setup
script to run, they aren't run here.
"""
from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from typing import Callable

log = logging.getLogger(__name__)

ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040
ES_CONTINUOUS = 0x80000000

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
_k32.SetThreadExecutionState.restype = wintypes.DWORD


def _set_state(flags: int) -> int:
    return int(_k32.SetThreadExecutionState(flags))


# Tests replace this with a recorder; production uses the real API.
set_state: Callable[[int], int] = _set_state


def keep_awake() -> bool:
    """Ask Windows not to sleep (system only; the display may still turn off)."""
    prev = set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if not prev:
        log.warning("SetThreadExecutionState(keep awake) failed")
        return False
    return True


def allow_sleep() -> bool:
    prev = set_state(ES_CONTINUOUS)
    return bool(prev)


def power_plan_commands() -> list[list[str]]:
    """powercfg commands for 'never sleep while plugged in' (setup step; not executed here)."""
    return [["powercfg", "/change", "standby-timeout-ac", "0"],
            ["powercfg", "/change", "hibernate-timeout-ac", "0"]]
