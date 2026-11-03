"""OpenCV colour/blob tracker behind the SegmentationBackend interface.

Development fallback (and a real-footage safety net): concepts are matched to a colour by
keyword ("orange pendulum bob" -> orange). Use brightly coloured objects when filming.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from ..schemas import Track, TrackFrame

# OpenCV HSV: H in [0,180)
COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "orange": [((5, 120, 120), (22, 255, 255))],
    "red": [((0, 130, 90), (5, 255, 255)), ((170, 130, 90), (180, 255, 255))],
    "yellow": [((22, 110, 120), (35, 255, 255))],
    "green": [((40, 80, 60), (85, 255, 255))],
    "blue": [((95, 120, 60), (130, 255, 255))],
    "pink": [((140, 80, 100), (170, 255, 255))],
    "magenta": [((140, 80, 100), (170, 255, 255))],
}
REFERENCE_WORDS = ("reference", "quarter", "coin", "card", "ruler", "paper", "sticker")


def color_for(concept: str) -> str:
    c = concept.lower()
    for name in COLOR_RANGES:
        if re.search(rf"\b{name}\b", c):
            return name
    return "green" if any(w in c for w in REFERENCE_WORDS) else "orange"


def _mask(hsv: np.ndarray, color: str) -> np.ndarray:
    m = None
    for lo, hi in COLOR_RANGES[color]:
        mm = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        m = mm if m is None else cv2.bitwise_or(m, mm)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


class FallbackTracker:
    name = "fallback_color_tracker"

    def __init__(self, min_area: int = 12):
        self.min_area = min_area

    def track(self, video_path: str, frame_times: list[float], concepts: list[str]) -> dict[str, Track]:
        colors = {c: color_for(c) for c in concepts}
        frames: dict[str, list[TrackFrame]] = {c: [] for c in concepts}
        prev: dict[str, tuple[float, float] | None] = {c: None for c in concepts}
        cap = cv2.VideoCapture(video_path)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_times[i] if i < len(frame_times) else (frame_times[-1] if frame_times else 0.0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            masks: dict[str, np.ndarray] = {}
            for c in concepts:
                col = colors[c]
                if col not in masks:
                    masks[col] = _mask(hsv, col)
                frames[c].append(self._detect(masks[col], hsv, i, t, prev[c]))
                f = frames[c][-1]
                prev[c] = f.centroid if f.score > 0 else prev[c]
            i += 1
        cap.release()
        return {c: Track(track_id=f"{colors[c]}:{c}", concept=c, frames=frames[c]) for c in concepts}

    def _detect(self, mask: np.ndarray, hsv: np.ndarray, i: int, t: float, prev) -> TrackFrame:
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cand = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= self.min_area]
        if not cand:
            return TrackFrame(frame_idx=i, t=t, bbox=(0, 0, 0, 0), score=0.0)
        areas = np.array([stats[k, cv2.CC_STAT_AREA] for k in cand], float)
        if prev is not None and len(cand) > 1:
            d = np.array([np.hypot(*(cents[k] - prev)) for k in cand])
            # prefer large blobs near the previous position
            k = cand[int(np.argmax(areas / areas.max() - d / (d.max() + 1e-6) * 0.5))]
        else:
            k = cand[int(np.argmax(areas))]
        x, y, w, h, area = stats[k]
        comp, x, y, w, h = self._refine(labels == k, hsv, x, y, w, h)
        area = int(comp.sum())
        if area == 0:
            return TrackFrame(frame_idx=i, t=t, bbox=(0, 0, 0, 0), score=0.0)
        m = cv2.moments(comp, binaryImage=True)
        cx, cy = x + m["m10"] / m["m00"], y + m["m01"] / m["m00"]
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        poly = cv2.approxPolyDP(cnt, 1.0, True).reshape(-1, 2) + np.array([x, y])
        score = float(min(1.0, area / (self.min_area * 3)))
        return TrackFrame(frame_idx=i, t=t, bbox=(float(x), float(y), float(w), float(h)), score=score,
                          polygon=[(float(a), float(b)) for a, b in poly], centroid=(float(cx), float(cy)),
                          area=float(area), min_rect=(float(max(rw, rh)), float(min(rw, rh))))

    @staticmethod
    def _refine(full_mask: np.ndarray, hsv: np.ndarray, x, y, w, h):
        """Re-threshold saturation at the half level between object and background.

        4:2:0 video blurs chroma; a fixed colour threshold makes masks ~1 px too large on each side,
        which biases scale calibration. A 50% threshold of a symmetric blur keeps edges in place.
        """
        H, W = full_mask.shape
        pad = 8
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
        comp = full_mask[y0:y1, x0:x1].astype(np.uint8)
        sat = hsv[y0:y1, x0:x1, 1].astype(np.float32)
        core = cv2.erode(comp, np.ones((3, 3), np.uint8))
        if core.sum() < 3:
            core = comp
        near = cv2.dilate(comp, np.ones((5, 5), np.uint8), iterations=1)
        far = cv2.dilate(comp, np.ones((5, 5), np.uint8), iterations=2) == 0
        if far.sum() < 10:
            return comp, x0, y0, x1 - x0, y1 - y0
        thr = (np.median(sat[core > 0]) + np.median(sat[far])) / 2
        ref = ((sat >= thr) & (near > 0)).astype(np.uint8)
        if ref.sum() == 0:
            ref = comp
        bx, by, bw, bh = _bbox(ref)
        return ref[by:by + bh, bx:bx + bw], x0 + bx, y0 + by, bw, bh

