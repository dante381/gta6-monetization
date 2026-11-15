"""Config loading: settings.yaml (tunable) merged with locked.yaml (protected).

Access with attributes:  cfg.scoring.gameplay_weights.audio_spike, cfg.locked.thresholds.publish
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from . import paths


class Node(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Node({k: copy.deepcopy(v, memo) for k, v in self.items()})


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Node({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(settings_path: Path | None = None, locked_path: Path | None = None,
                overrides: dict | None = None) -> Node:
    """Return the merged config. `overrides` (nested dict) is applied last (tests).

    The locked file is available as cfg.locked (and never merged into tunables).
    """
    sp = Path(settings_path or paths.SETTINGS_YAML)
    lp = Path(locked_path or paths.LOCKED_YAML)
    settings = yaml.safe_load(sp.read_text(encoding="utf-8")) or {}
    locked = yaml.safe_load(lp.read_text(encoding="utf-8")) or {}
    data = dict(settings)
    data["locked"] = locked
    if overrides:
        data = _merge(data, overrides)
    return _wrap(data)


def kill_switch_path(cfg: Node) -> Path:
    return paths.ROOT / cfg.locked.kill_switch_file


def paused(cfg: Node) -> bool:
    return kill_switch_path(cfg).exists()
