"""Filesystem layout. Everything lives inside the project folder (never on C:).

Tests (and tools) can redirect the data folder with CLIPPER_DATA_DIR.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TOOLS = ROOT / "tools"
FFMPEG_DIR = TOOLS / "ffmpeg" / "bin"
DENO_DIR = TOOLS / "deno"
VENV = ROOT / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"
TMP = ROOT / ".tmp"
CACHE = ROOT / ".cache"

CONFIG_DIR = ROOT / "config"
SETTINGS_YAML = CONFIG_DIR / "settings.yaml"
LOCKED_YAML = CONFIG_DIR / "locked.yaml"

PROMPTS = ROOT / "prompts"
SCHEMAS = PROMPTS / "schemas"

CLIENT_SECRET = ROOT / "client_secret.json"
SECRETS = ROOT / "secrets"
TOKEN = SECRETS / "token.json"

MUSIC = ROOT / "music"
HUD_REFS = ROOT / "hud_refs"
SITE = ROOT / "site"
KILL_SWITCH = ROOT / "PAUSE"  # name also in locked.yaml; ROOT/<name>


def data_dir() -> Path:
    return Path(os.environ.get("CLIPPER_DATA_DIR", ROOT / "data"))


class _DataPaths:
    """Lazy accessors so CLIPPER_DATA_DIR set by tests at runtime is honoured."""

    @property
    def root(self) -> Path:
        return data_dir()

    @property
    def chunks(self) -> Path:
        return data_dir() / "chunks"

    @property
    def frames(self) -> Path:
        return data_dir() / "frames"

    @property
    def grids(self) -> Path:
        return data_dir() / "grids"

    @property
    def work(self) -> Path:  # scratch for clip production (hook PNGs, concat lists ...)
        return data_dir() / "work"

    @property
    def ready(self) -> Path:
        return data_dir() / "clips" / "ready"

    @property
    def backup(self) -> Path:
        return data_dir() / "clips" / "backup"

    @property
    def logs(self) -> Path:
        return data_dir() / "logs"

    @property
    def heartbeat(self) -> Path:
        return data_dir() / "heartbeat"

    @property
    def samples(self) -> Path:
        return data_dir() / "samples"

    @property
    def reports(self) -> Path:
        return data_dir() / "reports"

    @property
    def claude_work(self) -> Path:
        return data_dir() / "claude_work"

    @property
    def state_db(self) -> Path:
        return data_dir() / "state.db"

    def ensure(self) -> None:
        for p in (self.chunks, self.frames, self.grids, self.work, self.ready, self.backup,
                  self.logs, self.heartbeat, self.samples, self.reports, self.claude_work):
            p.mkdir(parents=True, exist_ok=True)


DATA = _DataPaths()
