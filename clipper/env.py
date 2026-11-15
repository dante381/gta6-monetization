"""Environment for every subprocess: temp, caches and tool paths stay in the project."""
from __future__ import annotations

import os

from . import paths


def subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    tmp = str(paths.TMP)
    env.update({
        "TEMP": tmp,
        "TMP": tmp,
        "TMPDIR": tmp,
        "PIP_CACHE_DIR": str(paths.CACHE / "pip"),
        "DENO_DIR": str(paths.CACHE / "deno"),
        "XDG_CACHE_HOME": str(paths.CACHE),
        "PYTHONIOENCODING": "utf-8",
    })
    prepend = [str(paths.FFMPEG_DIR), str(paths.DENO_DIR), str(paths.VENV / "Scripts")]
    env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])
    if extra:
        env.update(extra)
    return env


def apply_to_current_process() -> None:
    """Make this Python process (and anything using tempfile) use project temp dirs."""
    for d in (paths.TMP, paths.CACHE):
        d.mkdir(parents=True, exist_ok=True)
    env = subprocess_env()
    for k in ("TEMP", "TMP", "TMPDIR", "PIP_CACHE_DIR", "DENO_DIR", "XDG_CACHE_HOME", "PATH"):
        os.environ[k] = env[k]
    import tempfile
    tempfile.tempdir = str(paths.TMP)
