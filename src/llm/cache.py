from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable

from src.utils.io import ensure_dir, write_json


class DiskCache:
    def __init__(self, directory: str | Path):
        self.directory = ensure_dir(directory)
        self._locks_guard = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def key(self, payload: dict[str, Any]) -> str:
        packed = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha256(packed).hexdigest()

    def get_or_call(self, payload: dict[str, Any], fn: Callable[[], dict[str, Any]]):
        key = self.key(payload)
        path = self.directory / f"{key}.json"
        with self._locks_guard:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                result["cached"] = True
                return result
            result = fn()
            write_json(path, result)
            result = dict(result)
            result["cached"] = False
            return result
