"""SAM 3.1 via the Meta Model API (Responses API, model `sam-3.1`).

Output format (dev.meta.ai/docs/sam/reading-segmentation): one line per frame,
  <Nf>ID<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H[;c=0.93]|><|mask;x=..;y=..;data=H,W,<enc>base85|>,ID<|box...
Boxes are XYXY source pixels. We request `include_confidence` and one-bit masks.

VERIFY (needs a working key): exact `input_video` content-part fields and the upload `purpose`;
the one-bit payload bit order. If mask decoding fails we fall back to box geometry.
"""
from __future__ import annotations

import base64
import re
from collections import defaultdict

import numpy as np

from ..config import get_settings
from ..schemas import Track, TrackFrame
from .cache import cached_call, file_sha256

FRAME_RE = re.compile(r"<(\d+)f>")
REC_RE = re.compile(r"(\d+)<\|box;([^|]*)\|>(?:<\|mask;([^|]*)\|>)?")


def _kv(s: str) -> dict[str, str]:
    out = {}
    for part in s.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def decode_mask(spec: str) -> tuple[np.ndarray, int, int] | None:
    """Return (binary mask HxW, x offset, y offset) or None if the payload can't be decoded."""
    kv = _kv(spec.split("data=", 1)[0])
    data = spec.split("data=", 1)[1] if "data=" in spec else ""
    try:
        h_s, w_s, rest = data.split(",", 2)
        h, w = int(h_s), int(w_s)
        enc, payload = rest[0], rest[1:]
        if enc != "!":  # lossless codec isn't documented publicly; use box geometry instead
            return None
        bits = np.unpackbits(np.frombuffer(base64.b85decode(payload), np.uint8))[: h * w]
        return bits.reshape(h, w).astype(np.uint8), int(float(kv.get("x", 0))), int(float(kv.get("y", 0)))
    except Exception:
        return None


def parse_output(text: str) -> dict[int, dict[int, dict]]:
    """{frame_idx: {object_id: {box, score, mask_spec}}}"""
    out: dict[int, dict[int, dict]] = defaultdict(dict)
    for line in text.splitlines():
        m = FRAME_RE.search(line)
        if not m:
            continue
        fi = int(m.group(1))
        for rec in REC_RE.finditer(line[m.end():]):
            oid, box_s, mask_s = int(rec.group(1)), _kv(rec.group(2)), rec.group(3)
            x1, y1, x2, y2 = (float(box_s.get(k, 0)) for k in ("x1", "y1", "x2", "y2"))
            out[fi][oid] = {"box": (x1, y1, x2, y2), "score": float(box_s.get("c", 1.0)), "mask": mask_s}
    return out


def _frame_from_record(fi: int, t: float, rec: dict) -> TrackFrame:
    x1, y1, x2, y2 = rec["box"]
    bw, bh = x2 - x1, y2 - y1
    cx, cy, area = (x1 + x2) / 2, (y1 + y2) / 2, np.pi / 4 * bw * bh
    long_s, short_s = max(bw, bh), min(bw, bh)
    dm = decode_mask(rec["mask"]) if rec.get("mask") else None
    if dm is not None and dm[0].any():
        mk, ox, oy = dm
        ys, xs = np.nonzero(mk)
        cx, cy, area = float(xs.mean() + ox), float(ys.mean() + oy), float(len(xs))
    return TrackFrame(frame_idx=fi, t=t, bbox=(x1, y1, bw, bh), score=rec["score"],
                      centroid=(cx, cy), area=area, min_rect=(long_s, short_s))


class MetaApiSam:
    name = "sam-3.1 (Meta Model API)"

    def __init__(self):
        self.s = get_settings()

    def _call(self, video_path: str, concept: str) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self.s.meta_base_url, api_key=self.s.model_api_key)
        with open(video_path, "rb") as f:
            up = client.files.create(file=f, purpose="user_data")  # VERIFY purpose
        stream = client.responses.create(
            model=self.s.sam_model,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": concept},
                {"type": "input_video", "file_id": up.id},  # VERIFY field name
            ]}],
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
            stream=True,
        )
        text = []
        for ev in stream:
            if getattr(ev, "type", "") == "response.output_text.delta":
                text.append(ev.delta)
        return "".join(text)

    def track(self, video_path: str, frame_times: list[float], concepts: list[str]) -> dict[str, Track]:
        vh = file_sha256(video_path)
        tracks = {}
        for concept in concepts:
            req = {"task": "video_segmentation", "concept": concept, "mask_encoding": "one_bit"}
            text = cached_call(self.s.sam_model, req, lambda: self._call(video_path, concept), [vh])
            per_frame = parse_output(text)
            # keep the object id with the best coverage x confidence
            score_by_id: dict[int, float] = defaultdict(float)
            for recs in per_frame.values():
                for oid, r in recs.items():
                    score_by_id[oid] += r["score"]
            best = max(score_by_id, key=score_by_id.get) if score_by_id else None
            frames = []
            for i, t in enumerate(frame_times):
                rec = per_frame.get(i, {}).get(best) if best is not None else None
                frames.append(_frame_from_record(i, t, rec) if rec else
                              TrackFrame(frame_idx=i, t=t, bbox=(0, 0, 0, 0), score=0.0))
            tracks[concept] = Track(track_id=f"sam:{best}:{concept}", concept=concept, frames=frames)
        return tracks
