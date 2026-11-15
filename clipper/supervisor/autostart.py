"""Windows Task Scheduler registration so the supervisor starts again after a reboot / logon.

    python -m clipper autostart register [--url URL] | unregister | status

The task (GTA6Clipper) runs at logon of the current user, without elevation (LeastPrivilege),
as `<.venv>\\Scripts\\pythonw.exe -m clipper supervise --url URL` with the project folder as the
working directory.

It is registered from an XML definition (written to .tmp/) because plain `schtasks /Create /TR`
can't set a working directory and gives every task a 72 hour run-time limit, which would kill a
supervisor that runs for days. If the XML route fails, the plain
`/SC ONLOGON /RL LIMITED /TR` command is tried; its /TR runs supervisor.py as a script so no
working directory is needed. Nothing is put in the Startup folder (it lives on C:). Errors are
reported, never hidden.
"""
from __future__ import annotations

import argparse
import getpass
import os
import subprocess
from pathlib import Path
from typing import Callable, Sequence
from xml.sax.saxutils import escape

from clipper import paths

TASK_NAME = "GTA6Clipper"

Runner = Callable[[Sequence[str]], tuple[int, str, str]]


class AutostartError(RuntimeError):
    pass


def default_runner(cmd: Sequence[str]) -> tuple[int, str, str]:
    p = subprocess.run(list(cmd), capture_output=True, timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    dec = lambda b: (b or b"").decode("utf-8", "replace") if b else ""  # noqa: E731
    return p.returncode, dec(p.stdout), dec(p.stderr)


def pythonw() -> Path:
    return paths.VENV / "Scripts" / "pythonw.exe"


def _url(url: str | None) -> str:
    if url:
        return url
    from clipper.config import load_config
    u = (load_config().get("stream") or {}).get("url") or ""
    if not u:
        raise AutostartError("no stream URL: pass --url or set stream.url in config/settings.yaml")
    return u


def supervise_args(url: str) -> list[str]:
    return ["-m", "clipper", "supervise", "--url", url]


def _q(s: str) -> str:
    return f'"{s}"'


def task_run_line(url: str) -> str:
    """Plain /TR command line (fallback): runs supervisor.py as a script (no cwd needed)."""
    script = paths.ROOT / "clipper" / "supervisor" / "supervisor.py"
    return f'{_q(str(pythonw()))} {_q(str(script))} --url {_q(url)}'


def task_xml(url: str, user: str | None = None) -> str:
    user = user or _current_user()
    args = " ".join(_q(a) if (" " in a or not a) else a for a in supervise_args(url))
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>GTA 6 Stream Auto Clipper supervisor (recorder + worker)</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{escape(user)}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{escape(user)}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT5M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(str(pythonw()))}</Command>
      <Arguments>{escape(args)}</Arguments>
      <WorkingDirectory>{escape(str(paths.ROOT))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _current_user() -> str:
    dom = os.environ.get("USERDOMAIN")
    name = os.environ.get("USERNAME") or getpass.getuser()
    return f"{dom}\\{name}" if dom else name


def xml_path() -> Path:
    return paths.TMP / f"{TASK_NAME}.task.xml"


def create_xml_cmd(xml_file: Path | str) -> list[str]:
    return ["schtasks", "/Create", "/F", "/TN", TASK_NAME, "/XML", str(xml_file)]


def create_simple_cmd(url: str) -> list[str]:
    return ["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/RL", "LIMITED", "/TN", TASK_NAME,
            "/TR", task_run_line(url)]


def delete_cmd() -> list[str]:
    return ["schtasks", "/Delete", "/F", "/TN", TASK_NAME]


def query_cmd() -> list[str]:
    return ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"]


def register(url: str | None = None, runner: Runner | None = None) -> dict:
    """Create (or replace) the logon task. Returns {"method", "output"}; raises AutostartError."""
    run = runner or default_runner
    url = _url(url)
    if not pythonw().exists():
        raise AutostartError(f"{pythonw()} not found (run scripts/setup.ps1 first)")
    xf = xml_path()
    xf.parent.mkdir(parents=True, exist_ok=True)
    xf.write_text(task_xml(url), encoding="utf-16")  # schtasks wants UTF-16 XML
    errors = []
    try:
        rc, out, err = run(create_xml_cmd(xf))
    finally:
        xf.unlink(missing_ok=True)
    if rc == 0:
        return {"method": "xml", "output": out.strip()}
    errors.append(f"XML: rc={rc} {(err or out).strip()}")
    rc, out, err = run(create_simple_cmd(url))
    if rc == 0:
        return {"method": "onlogon", "output": out.strip(),
                "warning": "registered with plain schtasks: Windows applies its default 72 h run-time limit"}
    errors.append(f"ONLOGON: rc={rc} {(err or out).strip()}")
    hint = ""
    if any("denied" in e.lower() for e in errors):
        hint = (" Access was denied: run `python -m clipper autostart register` once from an elevated "
                "(Administrator) PowerShell, or create the task in Task Scheduler by hand.")
    raise AutostartError("could not register the scheduled task. " + " | ".join(errors) + hint)


def unregister(runner: Runner | None = None) -> bool:
    rc, out, err = (runner or default_runner)(delete_cmd())
    if rc != 0 and "cannot find" not in (err + out).lower():
        raise AutostartError(f"schtasks /Delete failed: rc={rc} {(err or out).strip()}")
    return rc == 0


def status(runner: Runner | None = None) -> dict:
    """{"registered": bool, plus the fields of `schtasks /Query /V /FO LIST` (Status, Next Run Time...)}."""
    rc, out, err = (runner or default_runner)(query_cmd())
    if rc != 0:
        return {"registered": False, "error": (err or out).strip()}
    info: dict = {"registered": True}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            if k and k not in info:
                info[k] = v.strip()
    return info


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clipper autostart")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--url", default=None)
    sub.add_parser("unregister")
    sub.add_parser("status")
    args = ap.parse_args(list(argv) if argv is not None else None)
    try:
        if args.cmd == "register":
            res = register(args.url)
            print(f"registered {TASK_NAME} ({res['method']}). {res.get('warning', '')}".strip())
        elif args.cmd == "unregister":
            print("removed" if unregister() else "not registered")
        else:
            st = status()
            if not st["registered"]:
                print(f"{TASK_NAME}: not registered")
            else:
                for k in ("TaskName", "Status", "Next Run Time", "Last Run Time", "Last Result", "Task To Run"):
                    if k in st:
                        print(f"{k}: {st[k]}")
    except AutostartError as e:
        print(f"error: {e}")
        return 1
    return 0
