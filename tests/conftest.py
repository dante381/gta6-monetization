"""Shared test fixtures.

- Every test gets an isolated data dir (CLIPPER_DATA_DIR) inside .tmp/pytest.
- CLAUDE_BIN points at tests/fakes/fake_claude.py, so no real Claude usage in tests.
- `synthetic_chunk` / `synthetic_short` are cached real H.264 TS files with known moments
  (see tests/fixtures/make_synthetic.py for the timeline; truth dict alongside).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipper import env as _env  # noqa: E402

_env.apply_to_current_process()

FAKE_CLAUDE = ROOT / "tests" / "fakes" / "fake_claude.py"
FIXTURE_CACHE = ROOT / ".tmp" / "test_fixtures"


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(data))
    monkeypatch.setenv("CLAUDE_BIN", f'"{FAKE_CLAUDE}"')
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(tmp_path / "fake_claude.jsonl"))
    from clipper import paths
    paths.DATA.ensure()
    return data


@pytest.fixture
def cfg():
    from clipper.config import load_config
    return load_config()


@pytest.fixture
def state(isolated_data):
    from clipper.state import StateDB
    db = StateDB()
    db.set_current_stream("test-stream")
    yield db
    db.close()


@pytest.fixture
def fake_claude_calls(tmp_path):
    """Returns a function giving the list of logged fake claude calls."""
    def read():
        p = tmp_path / "fake_claude.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return read


@pytest.fixture
def claude_scenario(tmp_path, monkeypatch):
    """Call with a dict to set FAKE_CLAUDE_SCENARIO for this test."""
    def set_(scenario: dict):
        p = tmp_path / "scenario.json"
        p.write_text(json.dumps(scenario), encoding="utf-8")
        monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", str(p))
        return p
    return set_


def _cached_synthetic(name: str, duration: float):
    from tests.fixtures.make_synthetic import make
    FIXTURE_CACHE.mkdir(parents=True, exist_ok=True)
    out = FIXTURE_CACHE / f"{name}.ts"
    truth_p = out.with_suffix(".truth.json")
    if not (out.exists() and truth_p.exists()):
        make(out, duration)
    return out, json.loads(truth_p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def synthetic_chunk():
    """(path, truth) of a 300 s synthetic chunk."""
    return _cached_synthetic("synthetic_300", 300.0)


@pytest.fixture(scope="session")
def synthetic_short():
    """(path, truth) of a 60 s synthetic chunk (same timeline scaled by 0.2)."""
    return _cached_synthetic("synthetic_60", 60.0)
