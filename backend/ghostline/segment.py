"""Segmentation / tracking facade. Output: {concept: Track} normalized across backends."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import get_settings
from .schemas import Run, Track
from .storage import abs_path


class SegmentationBackend(Protocol):
    name: str

    def track(self, video_path: str, frame_times: list[float], concepts: list[str]) -> dict[str, Track]: ...


def get_backend(name: str | None = None) -> SegmentationBackend:
    name = name or get_settings().segment_backend
    if name == "meta":
        from .clients.sam_meta import MetaApiSam
        return MetaApiSam()
    if name == "local":
        from .clients.sam_local import LocalSam
        return LocalSam()
    from .clients.fallback_tracker import FallbackTracker
    return FallbackTracker()


def track_run(run: Run, concepts: list[str], backend: str | None = None) -> dict[str, Track]:
    be = get_backend(backend)
    tracks = be.track(str(abs_path(run.video_path)), run.frame_times, concepts)
    min_score = get_settings().sam_min_score
    for tr in tracks.values():
        for f in tr.frames:
            if f.score < min_score:
                f.centroid = None
    return tracks


def track_summary(tracks: dict[str, Track]) -> dict:
    out = {}
    for c, tr in tracks.items():
        valid = [f for f in tr.frames if f.centroid is not None]
        out[c] = {"track_id": tr.track_id, "frames": len(tr.frames),
                  "coverage": round(len(valid) / max(len(tr.frames), 1), 4),
                  "mean_score": round(float(np.mean([f.score for f in valid])) if valid else 0.0, 4)}
    return out
