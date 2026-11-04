"""Ghost overlay renderer (SPEC §14): each model's prediction drawn as a translucent ghost over the video."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import storage
from .measure import centroids
from .schemas import Run

COLORS = {  # BGR; same hex as frontend MODEL_COLORS (#2f6fdc, #d0306f, #c28a10)
    "pendulum_linear": (220, 111, 47),
    "pendulum_nonlinear": (111, 48, 208),
    "pendulum_nonlinear_quaddamp": (16, 138, 194),
    "drag_linear": (220, 111, 47),
    "drag_quadratic": (111, 48, 208),
}
LABELS = {
    "pendulum_linear": "linear (small-angle)",
    "pendulum_nonlinear": "nonlinear (sin)",
    "pendulum_nonlinear_quaddamp": "nonlinear + quad drag",
    "drag_linear": "linear drag",
    "drag_quadratic": "quadratic drag",
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALPHA = 0.42
TRAIL_S = 0.3


def _text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1, bg=None):
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    if bg is not None:
        x, y = org
        cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + base + 4), bg, -1)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)
    return w, h


class _Pipe:
    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, f):
        self.p.stdin.write(f.tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding overlay")


def _interp_band(band, t):
    if not band:
        return None
    bt = np.asarray(band["t"])
    return np.interp(t, bt, band["lo"]), np.interp(t, bt, band["hi"])


def render_run(run: Run, model_names: list[str] | None = None) -> Path:
    from .pipeline import load_curves

    curves = load_curves(run.id)
    system = run.scene.system_type if run.scene else "pendulum"
    ft = np.asarray(run.frame_times)
    invalid = run.validity is not None and run.validity.status == "invalid"
    models = [k for k in curves if k != "t" and (model_names is None or k in model_names)]
    if invalid:
        models = []
    ct = np.asarray(curves.get("t", ft))
    preds = {k: np.interp(ft, ct, curves[k]) for k in models}
    bands = {}
    if run.fits and run.role != "reveal":
        for f in run.fits:
            if f.model in preds and f.band:
                bands[f.model] = _interp_band(f.band, ft)

    ser = next(iter(run.series.values())) if run.series else None
    meas = np.interp(ft, ser.t, ser.y) if ser else None
    mvalid = np.asarray(ser.valid_mask, bool) if ser else np.zeros(len(ft), bool)
    cal = run.calibration
    obj_xy = None
    if run.tracks and run.scene:
        xy, v = centroids(run.tracks[run.scene.tracked_concepts[0]])
        if v.any():
            idx = np.arange(len(xy))
            obj_xy = np.column_stack([np.interp(idx, idx[v], xy[v, 0]), np.interp(idx, idx[v], xy[v, 1])])
    radius = (cal.object_radius_px if cal and cal.object_radius_px else 12.0)

    split_t = run.split.train_end_t if run.split else None
    is_reveal = run.role == "reveal"
    hash12 = None
    if is_reveal and run.commitment_id:
        try:
            hash12 = storage.load_commitment(run.commitment_id).sha256[:12]
        except KeyError:
            pass

    def to_px(k: str, i: int, val: float) -> tuple[float, float]:
        if system == "pendulum":
            th = val + cal.theta_offset
            return cal.pivot_px[0] + cal.length_px * np.sin(th), cal.pivot_px[1] + cal.length_px * np.cos(th)
        x = obj_xy[i, 0] if obj_xy is not None else run.width / 2
        return x, val / cal.meters_per_pixel

    out = storage.run_dir(run.id) / "overlay.mp4"
    cap = cv2.VideoCapture(str(storage.abs_path(run.video_path)))
    pipe = None
    sq_err = {k: [] for k in models}
    unit_scale, unit = (np.degrees(1), "deg") if system == "pendulum" else (100.0, "cm")
    trail_n = max(int(TRAIL_S * (run.fps_nominal or 30)), 1)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(ft):
            break
        h, w = frame.shape[:2]
        if pipe is None:
            pipe = _Pipe(out, w, h, run.fps_nominal or 30)
        t = ft[i]
        over = frame.copy()
        for k in models:
            col = COLORS.get(k, (255, 255, 255))
            if k in bands and bands[k] is not None and system == "pendulum":
                lo, hi = bands[k][0][i], bands[k][1][i]
                arc = [to_px(k, i, a) for a in np.linspace(lo, hi, 8)]
                pts = np.array(arc, np.int32).reshape(-1, 1, 2)
                cv2.polylines(over, [pts], False, col, int(radius * 2.6), cv2.LINE_AA)
            for j in range(max(0, i - trail_n), i):
                px, py = to_px(k, j, preds[k][j])
                fade = (j - (i - trail_n)) / trail_n
                cv2.circle(over, (int(px), int(py)), max(int(radius * 0.35 * fade), 1), col, -1, cv2.LINE_AA)
            px, py = to_px(k, i, preds[k][i])
            if system == "pendulum":
                cv2.line(over, (int(cal.pivot_px[0]), int(cal.pivot_px[1])), (int(px), int(py)), col, 2, cv2.LINE_AA)
            cv2.circle(over, (int(px), int(py)), int(radius), col, -1, cv2.LINE_AA)
        frame = cv2.addWeighted(over, ALPHA, frame, 1 - ALPHA, 0)
        for k in models:  # crisp outline on top
            px, py = to_px(k, i, preds[k][i])
            cv2.circle(frame, (int(px), int(py)), int(radius), COLORS.get(k, (255, 255, 255)), 2, cv2.LINE_AA)

        # running RMSE after the prediction boundary
        boundary = split_t if split_t is not None else 0
        if meas is not None and t > boundary and mvalid[i]:
            for k in models:
                sq_err[k].append((preds[k][i] - meas[i]) ** 2)

        # legend
        y0 = 30
        title = "GHOSTLINE" + ("  |  REVEAL" if is_reveal else "")
        _text(frame, title, (16, y0), 0.65, (255, 255, 255), 2, bg=(30, 30, 30))
        for n, k in enumerate(models):
            y = y0 + 34 + 28 * n
            col = COLORS.get(k, (255, 255, 255))
            cv2.circle(frame, (24, y - 6), 8, col, -1, cv2.LINE_AA)
            err = (f"RMSE {np.sqrt(np.mean(sq_err[k])) * unit_scale:.2f} {unit}" if sq_err[k] else "fitting window")
            _text(frame, f"{LABELS.get(k, k)}   {err}", (40, y), 0.52, (255, 255, 255), 1, bg=(30, 30, 30))

        if hash12:
            _text(frame, f"COMMITTED {hash12}  VERIFIED", (w - 360, 30), 0.55, (120, 255, 160), 1, bg=(30, 30, 30))
        if invalid:
            reasons = run.validity.reasons[:2]
            _text(frame, "INVALID RUN - NOT SCORED", (16, h - 70), 0.8, (255, 255, 255), 2, bg=(30, 30, 200))
            for n, rs in enumerate(reasons):
                _text(frame, rs[:95], (16, h - 38 + 22 * n), 0.45, (255, 255, 255), 1, bg=(30, 30, 30))

        # timeline bar with split marker
        dur = max(ft[-1], 1e-6)
        cv2.rectangle(frame, (0, h - 8), (w, h), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h - 8), (int(w * t / dur), h), (220, 220, 220), -1)
        if split_t is not None:
            sx = int(w * split_t / dur)
            cv2.line(frame, (sx, h - 22), (sx, h), (0, 220, 255), 3)
            if not invalid and split_t <= t < split_t + 1.5:
                label = "PURE PREDICTION STARTS HERE" if is_reveal else "PREDICTION STARTS HERE"
                _text(frame, label, (w // 2 - 190, h - 40), 0.75, (0, 0, 0), 2, bg=(0, 220, 255))
            elif not invalid and t < split_t:
                _text(frame, "conditioning window" if is_reveal else "models see this part",
                      (16, h - 22), 0.5, (230, 230, 230), 1, bg=(30, 30, 30))
            elif not invalid:
                _text(frame, "prediction (models never saw this)", (16, h - 22), 0.5, (230, 230, 230), 1, bg=(30, 30, 30))
        pipe.write(frame)
        i += 1
    cap.release()
    if pipe:
        pipe.close()
    return out
