"""Synthetic video generator: pendulum / falling filter with known physics.

Renders a bright orange object on a textured wall plus a green reference object of known size,
with per-frame tracking jitter, pixel noise and optional camera bump / zoom (for validity tests).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .models.base import get_model

ORANGE = (20, 130, 250)  # BGR
ORANGE_DARK = (10, 90, 200)
GREEN = (60, 175, 50)
STRING = (50, 50, 55)
SHIFT = 4
SCALE = 1 << SHIFT
# OpenCV anti-aliased fills come out larger than requested; compensate so ground truth sizes are exact.
AA_CIRCLE = 0.55
AA_RECT = 0.7

QUARTER_M = 0.02426
CARD_M = 0.0856


def _p(x: float, y: float) -> tuple[int, int]:
    return int(round(x * SCALE)), int(round(y * SCALE))


def _background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    base = np.empty((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base[..., 0] = 196 + 10 * xx / w
    base[..., 1] = 200 + 6 * yy / h
    base[..., 2] = 204 - 8 * xx / w
    blob = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 25)
    blob = blob / (np.abs(blob).max() + 1e-6) * 9
    base += blob[..., None]
    grain = cv2.GaussianBlur(rng.normal(0, 3, (h, w)).astype(np.float32), (0, 0), 1.2)
    base += grain[..., None]
    return np.clip(base, 0, 255).astype(np.uint8)


def _camera(frame: np.ndarray, t: float, cfg) -> np.ndarray:
    h, w = frame.shape[:2]
    M = np.eye(3)
    if cfg.bump_at is not None and t >= cfg.bump_at:
        M = np.array([[1, 0, cfg.bump_px[0]], [0, 1, cfg.bump_px[1]], [0, 0, 1]], float) @ M
    if cfg.zoom_at is not None and t >= cfg.zoom_at:
        z = cfg.zoom
        c = np.array([[z, 0, (1 - z) * w / 2], [0, z, (1 - z) * h / 2], [0, 0, 1]], float)
        M = c @ M
    if np.allclose(M, np.eye(3)):
        return frame
    return cv2.warpAffine(frame, M[:2], (w, h), borderMode=cv2.BORDER_REPLICATE)


class _Writer:
    def __init__(self, path: Path, w: int, h: int, fps: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.p.stdin.write(frame.tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing synthetic video")


@dataclass
class SynthConfig:
    system: str = "pendulum"  # pendulum | falling_drag
    truth_model: str = ""  # default: pendulum_nonlinear / drag_quadratic
    fps: float = 60.0
    duration: Optional[float] = None
    track_noise_px: float = 0.3
    pixel_noise: float = 3.0
    seed: int = 0
    bump_at: Optional[float] = None
    bump_px: tuple[float, float] = (30.0, 12.0)
    zoom_at: Optional[float] = None
    zoom: float = 1.08
    # pendulum
    angle_deg: float = 8.0
    length_m: float = 0.6
    gamma: float = 0.03
    # drag
    n_filters: int = 1
    vt1: float = 1.1
    release_above_m: float = 0.3
    extra: dict = field(default_factory=dict)


def _noise_bank(shape, sigma, rng, n=6):
    if sigma <= 0:
        return [np.zeros(shape, np.int16)]
    return [np.round(rng.normal(0, sigma, shape)).astype(np.int16) for _ in range(n)]


def render_pendulum(out: Path, cfg: SynthConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    w, h, ppm = 1280, 720, 800.0
    pivot = (640.0, 70.0)
    L_px = cfg.length_m * ppm
    bob_r = 0.025 * ppm
    ref_c, ref_r = (1185.0, 650.0), QUARTER_M * ppm / 2
    duration = cfg.duration or 12.0
    model = get_model(cfg.truth_model or "pendulum_nonlinear")
    phys = {"omega0": float(np.sqrt(9.81 / cfg.length_m)), "gamma": cfg.gamma}
    t = np.arange(int(round(duration * cfg.fps))) / cfg.fps
    theta = model.simulate(t, phys, [np.deg2rad(cfg.angle_deg), 0.0], 0.0)

    bg = _background(w, h, rng)
    noise = _noise_bank((h, w, 3), cfg.pixel_noise, rng)
    wr = _Writer(out, w, h, cfg.fps)
    for i, (ti, th) in enumerate(zip(t, theta)):
        f = bg.copy()
        jx, jy = rng.normal(0, cfg.track_noise_px, 2)
        bx = pivot[0] + L_px * np.sin(th) + jx
        by = pivot[1] + L_px * np.cos(th) + jy
        cv2.rectangle(f, (int(pivot[0]) - 40, int(pivot[1]) - 14), (int(pivot[0]) + 40, int(pivot[1]) - 4), (90, 80, 70), -1)
        cv2.line(f, _p(*pivot), _p(bx, by), STRING, 2, cv2.LINE_AA, SHIFT)
        cv2.circle(f, _p(bx, by), int((bob_r - AA_CIRCLE) * SCALE), ORANGE, -1, cv2.LINE_AA, SHIFT)
        cv2.circle(f, _p(bx - bob_r * 0.3, by - bob_r * 0.3), int(bob_r * 0.25 * SCALE), (90, 190, 255), -1, cv2.LINE_AA, SHIFT)
        cv2.circle(f, _p(*ref_c), int((ref_r - AA_CIRCLE) * SCALE), GREEN, -1, cv2.LINE_AA, SHIFT)
        f = _camera(f, ti, cfg)
        f = np.clip(f.astype(np.int16) + noise[i % len(noise)], 0, 255).astype(np.uint8)
        wr.write(f)
    wr.close()
    return {"system": "pendulum", "truth_model": model.name, "params": phys,
            "x0": [float(np.deg2rad(cfg.angle_deg)), 0.0], "knobs": {"release_angle_deg": cfg.angle_deg},
            "px_per_m": ppm, "pivot_px": pivot, "length_px": L_px, "length_m": cfg.length_m,
            "reference": {"concept": "green reference disk (US quarter size)", "known_size_m": QUARTER_M, "dimension": "diameter"},
            "fps": cfg.fps, "duration": duration, "config": asdict(cfg)}


def render_drag(out: Path, cfg: SynthConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    w, h, ppm = 720, 1280, 550.0
    model = get_model(cfg.truth_model or "drag_quadratic")
    vt = cfg.vt1 * (cfg.n_filters ** model.exponent)
    phys = {"v_t": vt}
    y0 = -cfg.release_above_m
    # time to leave the frame + 0.3 s
    tt = np.linspace(0, 15, 6001)
    yy = model.simulate(tt, phys, [y0, 0.0], 0.0)
    exit_t = float(tt[min(np.searchsorted(yy, h / ppm + 0.1), len(tt) - 1)])
    duration = cfg.duration or exit_t + 0.3
    t = np.arange(int(round(duration * cfg.fps))) / cfg.fps
    y = model.simulate(t, phys, [y0, 0.0], 0.0)
    fw = 0.12 * ppm / 2
    fh = (0.012 + 0.004 * cfg.n_filters) * ppm / 2
    card = (600.0, 1160.0, CARD_M * ppm, 0.05398 * ppm)

    bg = _background(w, h, rng)
    noise = _noise_bank((h, w, 3), cfg.pixel_noise, rng)
    wr = _Writer(out, w, h, cfg.fps)
    for i, (ti, yi) in enumerate(zip(t, y)):
        f = bg.copy()
        jx, jy = rng.normal(0, cfg.track_noise_px, 2)
        cx = w / 2 + 0.03 * ppm * np.sin(2 * np.pi * 1.3 * ti) + jx
        cy = yi * ppm + jy
        cx0, cy0, cw, ch = card
        cv2.rectangle(f, _p(cx0 - cw / 2 + AA_RECT, cy0 - ch / 2 + AA_RECT), _p(cx0 + cw / 2 - AA_RECT, cy0 + ch / 2 - AA_RECT), GREEN, -1, cv2.LINE_AA, SHIFT)
        if -fh * 2 < cy < h + fh * 2:
            ang = 6 * np.sin(2 * np.pi * 0.9 * ti)
            cv2.ellipse(f, _p(cx, cy), (int(fw * SCALE), int(fh * SCALE)), ang, 0, 360, ORANGE, -1, cv2.LINE_AA, SHIFT)
            cv2.ellipse(f, _p(cx, cy - fh * 0.4), (int(fw * 0.9 * SCALE), int(fh * 0.35 * SCALE)), ang, 0, 360, ORANGE_DARK, -1, cv2.LINE_AA, SHIFT)
        f = _camera(f, ti, cfg)
        f = np.clip(f.astype(np.int16) + noise[i % len(noise)], 0, 255).astype(np.uint8)
        wr.write(f)
    wr.close()
    return {"system": "falling_drag", "truth_model": model.name, "params": phys, "vt1": cfg.vt1,
            "x0": [y0, 0.0], "knobs": {"n_filters": float(cfg.n_filters)}, "px_per_m": ppm,
            "reference": {"concept": "green reference card (credit card size)", "known_size_m": CARD_M, "dimension": "long_side"},
            "fps": cfg.fps, "duration": duration, "config": asdict(cfg)}


def make(out: Path, cfg: SynthConfig) -> dict:
    out = Path(out)
    truth = render_pendulum(out, cfg) if cfg.system == "pendulum" else render_drag(out, cfg)
    out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=1))
    return truth
