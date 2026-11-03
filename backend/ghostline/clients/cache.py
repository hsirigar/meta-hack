"""Disk cache for every external call (SAM, LLM). Key = sha256(model + request_json + file_hashes)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from ..config import get_settings
from ..storage import data_dir


class CacheMiss(RuntimeError):
    """Raised in DEMO_MODE when a call is not cached (no network allowed)."""


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(model: str, request: Any, file_hashes: list[str] | None = None) -> str:
    blob = json.dumps({"model": model, "request": request, "files": sorted(file_hashes or [])},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def cached_call(model: str, request: Any, fn: Callable[[], Any], file_hashes: list[str] | None = None) -> Any:
    key = cache_key(model, request, file_hashes)
    d = data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())["response"]
    if get_settings().demo_mode:
        raise CacheMiss(f"DEMO_MODE: no cached response for {model} ({key[:12]})")
    resp = fn()
    p.write_text(json.dumps({"model": model, "request": request, "response": resp}, default=str))
    return resp
