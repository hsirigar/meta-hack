"""SAM 3.1 via the Meta Model API (Responses API, model `sam-3.1`).

Output format (dev.meta.ai/docs/sam/reading-segmentation): one line per frame,
  <Nf>ID<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H[;c=0.93]|><|mask;x=..;y=..;data=H,W,<enc>base85|>,ID<|box...
Boxes are XYXY source pixels. We request `include_confidence` and one-bit masks.

Verified against the live API (Nov 4): upload with purpose="user_data", then an `input_video` part with
`file_id`; the output arrives in `response.output_text.done` (no deltas) and covers every source frame.
The mask payload uses a compressed codec that is not publicly documented (Meta's `meta-sam-parser`
package decodes it), so we measure from the box: centre and width of a compact object are what we need.

SAM 3.1 matches short, concrete noun phrases ("orange ball") far better than descriptive ones
("orange pendulum bob"), so `_pick_prompt` probes a few short variants on one frame first.
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
        if enc != "!":
            return None
        raw = base64.b85decode(payload)
        if len(raw) * 8 < h * w or len(raw) > (h * w + 7) // 8 + 4:  # compressed payload: not raw bits
            return None
        bits = np.unpackbits(np.frombuffer(raw, np.uint8))[: h * w]
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


SHAPES = ["ball", "circle", "disk", "ellipse", "rectangle", "strip", "object"]
COLOR_WORDS = ("white", "black", "grey", "gray", "silver", "brown", "purple", "orange", "red", "yellow",
               "green", "blue", "pink", "magenta")


def prompt_variants(concept: str) -> list[str]:
    """The concept itself, its last two words, then colour + simple shape nouns."""
    from .fallback_tracker import color_for
    words = concept.replace("(", " ").replace(")", " ").split()
    out = [concept]
    if len(words) > 2:
        out.append(" ".join(words[-2:]))
    named = [w for w in (x.lower() for x in words) if w in COLOR_WORDS]
    color = named[0] if named else color_for(concept)
    out += [f"{color} {s}" for s in SHAPES]
    seen, uniq = set(), []
    for p in out:
        if p.lower() not in seen:
            seen.add(p.lower()); uniq.append(p)
    return uniq


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

    def _client(self):
        from openai import OpenAI
        return OpenAI(base_url=self.s.meta_base_url, api_key=self.s.model_api_key, timeout=600)

    def _image_call(self, jpeg_b64: str, prompt: str) -> str:
        r = self._client().responses.create(model=self.s.sam_model, input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{jpeg_b64}"}]}],
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"})
        return r.output_text or ""

    def _probe_order(self, video_path: str, vh: str, concept: str) -> list[str]:
        """Prompt variants ordered by how many probe frames (25/50/75% of the clip) SAM finds them in."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frames = []
        for fi in (n // 4, n // 2, 3 * n // 4):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if ok:
                frames.append((fi, base64.b64encode(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()).decode()))
        cap.release()
        hits = {}
        for p in prompt_variants(concept):
            k = 0
            for fi, b64 in frames:
                req = {"task": "image_probe", "prompt": p, "frame": fi}
                k += bool(parse_output(cached_call(self.s.sam_model, req, lambda: self._image_call(b64, p), [vh])))
            hits[p] = k
            if k == len(frames):
                break  # found everywhere: good enough
        ordered = sorted(hits, key=lambda p: -hits[p])
        return ordered + [p for p in prompt_variants(concept) if p not in hits]

    def _call(self, video_path: str, concept: str) -> str:
        client = self._client()
        with open(video_path, "rb") as f:
            up = client.files.create(file=f, purpose="user_data")
        stream = client.responses.create(
            model=self.s.sam_model,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": concept},
                {"type": "input_video", "file_id": up.id},
            ]}],
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
            stream=True,
        )
        deltas, done = [], None
        for ev in stream:
            et = getattr(ev, "type", "")
            if et == "response.output_text.delta":
                deltas.append(ev.delta)
            elif et == "response.output_text.done":
                done = ev.text
            elif et in ("error", "response.failed"):
                raise RuntimeError(f"SAM 3.1 request failed: {ev}")
        return done if done is not None else "".join(deltas)

    def track(self, video_path: str, frame_times: list[float], concepts: list[str]) -> dict[str, Track]:
        vh = file_sha256(video_path)
        tracks = {}
        n = max(len(frame_times), 1)
        for concept in concepts:
            best_try = None
            for prompt in self._probe_order(video_path, vh, concept)[:5]:
                req = {"task": "video_segmentation", "concept": prompt, "mask_encoding": "one_bit"}
                per_frame = parse_output(cached_call(self.s.sam_model, req, lambda: self._call(video_path, prompt), [vh]))
                if best_try is None or len(per_frame) > len(best_try[1]):
                    best_try = (prompt, per_frame)
                if len(per_frame) >= 0.5 * n:
                    break
            prompt, per_frame = best_try
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
            tracks[concept] = Track(track_id=f"sam3.1:{best}:{prompt}", concept=concept, frames=frames)
        return tracks
