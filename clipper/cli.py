"""Command line: python -m clipper <command> [args]

  check        pre-stream checks (GPU, tools, claude login, secrets)
  record       recorder process        (normally started by the supervisor)
  work         processing worker       (normally started by the supervisor)
  supervise    supervisor: starts + restarts recorder and worker  (the one you run)
  health       show problems now; --watch prints one line per new problem (also runs inside supervise)
  auth         one-time YouTube sign-in
  publish      publish whatever is still queued (honours PAUSE / test mode), then exit
  report       post-stream report
  calibrate    export / summarize calibration review
  autostart    register / unregister / status of the Windows scheduled task
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    from .env import apply_to_current_process
    apply_to_current_process()
    if cmd == "check":
        from .startup_check import main as m
    elif cmd == "record":
        from .recorder.recorder import main as m
    elif cmd == "work":
        from .worker import main as m
    elif cmd == "publish":
        from .publish.publisher import main as m
    elif cmd == "health":
        from .health import main as m
    elif cmd == "supervise":
        from .supervisor.supervisor import main as m
    elif cmd == "auth":
        from .publish.auth import main as m
    elif cmd == "report":
        from .report import main as m
    elif cmd == "calibrate":
        from .calibrate import main as m
    elif cmd == "autostart":
        from .supervisor.autostart import main as m
    else:
        print(f"unknown command: {cmd}\n{__doc__}")
        return 2
    return int(m(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
