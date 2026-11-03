"""Local SAM 3.1 (open weights, `facebook/sam3.1`) on a Modal GPU function.

Optional fallback. Requires `modal` and a deployed function named `ghostline-sam3`
that accepts (video_bytes, concepts) and returns the same per-frame record format as
`sam_meta.parse_output`. Not needed for the synthetic/demo path.
"""
from __future__ import annotations

from ..schemas import Track, TrackFrame
from .cache import cached_call, file_sha256
from .sam_meta import _frame_from_record


class LocalSam:
    name = "sam-3.1 (local open weights)"

    def track(self, video_path: str, frame_times: list[float], concepts: list[str]) -> dict[str, Track]:
        try:
            import modal  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("SEGMENT_BACKEND=local needs `pip install modal` and a deployed ghostline-sam3 app") from e
        fn = modal.Function.from_name("ghostline-sam3", "segment_video")
        vh = file_sha256(video_path)
        video = open(video_path, "rb").read()
        out = {}
        for c in concepts:
            recs = cached_call("sam3.1-local", {"concept": c}, lambda: fn.remote(video, [c])[c], [vh])
            frames = []
            for i, t in enumerate(frame_times):
                r = recs.get(str(i))
                frames.append(_frame_from_record(i, t, r) if r else TrackFrame(frame_idx=i, t=t, bbox=(0, 0, 0, 0), score=0.0))
            out[c] = Track(track_id=f"sam-local:{c}", concept=c, frames=frames)
        return out
