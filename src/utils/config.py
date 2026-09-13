from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .io import load_yaml


def load_config(path: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(load_yaml(path))
    for dotted_key, value in (overrides or {}).items():
        if value is None:
            continue
        cursor = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg

