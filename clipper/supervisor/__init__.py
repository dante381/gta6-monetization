"""Supervisor: job object (2 GB cap), restarts, hang detection, repair agent, autostart, keep-awake.

Entry point: `python -m clipper supervise` -> clipper.supervisor.supervisor.main
"""


def main(argv=None) -> int:
    from .supervisor import main as _main
    return _main(argv)
