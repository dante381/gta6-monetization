"""Best-effort Windows desktop toasts (built-in WinRT API via PowerShell; no installs, errors swallowed)."""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

log = logging.getLogger("clipper.notify")

_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
_PS = (
    "[void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime];"
    "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
    "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
    "$n=$t.GetElementsByTagName('text');"
    "[void]$n.Item(0).AppendChild($t.CreateTextNode($env:CLIPPER_TOAST_TITLE));"
    "[void]$n.Item(1).AppendChild($t.CreateTextNode($env:CLIPPER_TOAST_MSG));"
    "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('" + _APP_ID + "')"
    ".Show([Windows.UI.Notifications.ToastNotification]::new($t))"
)

_last_toast = 0.0
_lock = threading.Lock()


def _run_powershell(title: str, message: str, timeout: float = 10.0) -> None:
    from clipper.env import subprocess_env
    env = dict(subprocess_env())
    env["CLIPPER_TOAST_TITLE"], env["CLIPPER_TOAST_MSG"] = title[:100], message[:250]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run(["powershell.exe", "-NoProfile", "-WindowStyle", "Hidden", "-Command", _PS], env=env,
                   timeout=timeout, creationflags=flags, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def toast(title: str, message: str, *, min_gap_s: float = 60.0, wait: bool = False) -> bool:
    """Show a toast unless one was shown within `min_gap_s`. Returns True if one was attempted."""
    global _last_toast
    if os.name != "nt":
        return False
    with _lock:
        now = time.monotonic()
        if _last_toast and now - _last_toast < min_gap_s:
            return False
        _last_toast = now

    def go():
        try:
            _run_powershell(title, message)
        except Exception as e:  # noqa: BLE001
            log.debug("toast failed: %s", e)

    if wait:
        go()
    else:
        threading.Thread(target=go, name="toast", daemon=True).start()
    return True


def notify(finding, min_gap_s: float = 60.0, wait: bool = False) -> None:
    """critical -> toast + log; warn -> toast; info -> log only."""
    text = f"{finding.code}: {finding.message}"
    if finding.level == "critical":
        log.error("HEALTH CRITICAL %s", text)
    elif finding.level == "warn":
        log.warning("HEALTH WARN %s", text)
    else:
        log.info("HEALTH INFO %s", text)
        return
    toast(f"GTA6 clipper: {finding.level.upper()}", text, min_gap_s=min_gap_s, wait=wait)


def make_notifier(cfg):
    from clipper.health import hcfg
    gap = float(hcfg(cfg, "toast_min_gap_s"))
    return lambda finding: notify(finding, min_gap_s=gap)
