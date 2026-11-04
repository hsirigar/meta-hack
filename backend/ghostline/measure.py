"""Pixels -> calibrated physical time series (SPEC §7.4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_coeffs, savgol_filter

from .schemas import Calibration, ReferenceObject, Run, TimeSeries, Track


# ---------- track helpers ----------
def centroids(track: Track) -> tuple[np.ndarray, np.ndarray]:
    """(n,2) centroids (nan where invalid), valid mask."""
    xy = np.full((len(track.frames), 2), np.nan)
    for i, f in enumerate(track.frames):
        if f.centroid is not None:
            xy[i] = f.centroid
    return xy, ~np.isnan(xy[:, 0])


def reference_sizes_px(track: Track, ref: ReferenceObject) -> np.ndarray:
    out = np.full(len(track.frames), np.nan)
    for i, f in enumerate(track.frames):
        if f.centroid is None:
            continue
        if ref.dimension == "diameter" and f.area:
            out[i] = 2 * np.sqrt(f.area / np.pi)
        elif ref.dimension == "long_side" and f.min_rect:
            out[i] = f.min_rect[0]
        elif ref.dimension == "width":
            out[i] = f.bbox[2]
        elif ref.dimension == "height":
            out[i] = f.bbox[3]
        elif f.min_rect:
            out[i] = f.min_rect[0]
    return out


def robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def reference_scale(track: Track, ref: ReferenceObject) -> tuple[float, float]:
    sizes = reference_sizes_px(track, ref)
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    if len(sizes) == 0:
        raise ValueError(f"reference object '{ref.concept}' was never detected; cannot calibrate scale")
    mpp_frames = ref.known_size_m / sizes
    mpp = float(np.median(mpp_frames))
    return mpp, float(max(robust_std(mpp_frames), 1e-12))


# ---------- circle fit (pendulum pivot) ----------
def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Kåsa algebraic fit, refined by geometric least squares. Returns (cx, cy, R, rms residual)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    R = np.sqrt(max(c[2] + cx ** 2 + cy ** 2, 1e-9))

    def res(p):
        return np.hypot(x - p[0], y - p[1]) - p[2]

    sol = least_squares(res, [cx, cy, R], method="lm")
    cx, cy, R = sol.x
    return float(cx), float(cy), float(abs(R)), float(np.sqrt(np.mean(sol.fun ** 2)))


def fit_circle_pivot_above(x, y, frame_h: float) -> tuple[float, float, float, float]:
    cx, cy, R, rms = fit_circle(x, y)
    if cy > np.nanmedian(y):  # fitted the wrong side (degenerate short arc): mirror the solution
        cy = 2 * np.nanmedian(y) - cy
    return cx, cy, R, rms


# ---------- smoothing / noise ----------
def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def smooth_and_noise(t: np.ndarray, y: np.ndarray, valid: np.ndarray, window_s: float = 0.1,
                     polyorder: int = 3) -> tuple[np.ndarray, float, np.ndarray]:
    """Savitzky-Golay smooth (display/derivatives only) + robust noise sigma + updated valid mask."""
    valid = valid.copy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    win = max(_odd(int(round(window_s / dt))), polyorder + 2 + (polyorder % 2 == 0))
    win = _odd(win)
    h0 = float(savgol_coeffs(win, polyorder)[win // 2])
    noise, ys = 0.0, y.copy()
    for _ in range(2):  # second pass after outlier rejection
        if valid.sum() < win + 2:
            return y.copy(), float("nan"), valid
        yi = np.interp(t, t[valid], y[valid])
        ys = savgol_filter(yi, win, polyorder)
        r = (y - ys)[valid]
        noise = max(robust_std(r) / np.sqrt(max(1 - h0, 1e-3)), 1e-9)
        bad = valid & (np.abs(y - ys) > 5 * noise)
        if not bad.any():
            break
        valid &= ~bad
    return ys, float(noise), valid


def trim_stationary(t, y, valid, frac: float = 0.15) -> np.ndarray:
    """Drag runs: drop leading/trailing frames where the object is not moving (held / landed)."""
    valid = valid.copy()
    if valid.sum() < 6:
        return valid
    idx = np.where(valid)[0]
    v = np.gradient(y[idx], t[idx])
    vmed = np.median(np.abs(v))
    moving = np.abs(v) > frac * vmed
    if not moving.any():
        return valid
    first, last = np.argmax(moving), len(moving) - 1 - np.argmax(moving[::-1])
    valid[idx[:first]] = False
    valid[idx[last + 1:]] = False
    return valid


def oscillation_center(t: np.ndarray, y: np.ndarray) -> float:
    """Equilibrium of a (damped) oscillation: time-mean between the first and last upward crossing."""
    c = float(np.median(y))
    for _ in range(3):
        s = np.sign(y - c)
        up = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
        if len(up) < 2:
            return c
        a, b = up[0], up[-1] + 1
        c_new = float(np.trapezoid(y[a:b + 1], t[a:b + 1]) / (t[b] - t[a]))
        if abs(c_new - c) < 1e-9:
            break
        c = c_new
    return c


@dataclass
class Measurement:
    calibration: Calibration
    series: dict[str, TimeSeries]
    primary: str


def _object_radius(track: Track) -> float:
    areas = [f.area for f in track.frames if f.centroid is not None and f.area]
    return float(np.sqrt(np.median(areas) / np.pi)) if areas else 10.0


def apply_homography(H: list[list[float]], xy: np.ndarray) -> np.ndarray:
    Hm = np.asarray(H, float)
    pts = np.column_stack([xy, np.ones(len(xy))]) @ Hm.T
    return pts[:, :2] / pts[:, 2:3]


def measure_run(run: Run, system_type: str, homography: list[list[float]] | None = None) -> Measurement:
    if run.scene is None or run.tracks is None:
        raise ValueError("run needs a scene description and tracks before measurement")
    obj_concept = run.scene.tracked_concepts[0]
    ref = run.scene.reference_object
    obj = run.tracks[obj_concept]
    t = np.asarray(run.frame_times[: len(obj.frames)], float)
    xy, valid = centroids(obj)

    if homography is not None:
        # homography maps image px -> plane coordinates in metres (x right, y down)
        good = valid.copy()
        xy[good] = apply_homography(homography, xy[good])
        mpp, mpp_std, method = 1.0, 0.0, "ruler_points"
    elif ref.known_size_m <= 0:
        if system_type != "pendulum":
            raise ValueError("a reference object of known size is needed to measure a fall in metres")
        mpp, mpp_std, method = 0.0, 0.0, "angles_only"  # pendulum angle and period need no scale
    else:
        mpp, mpp_std = reference_scale(run.tracks[ref.concept], ref)
        method = "reference_object"

    if system_type == "pendulum":
        x, y = xy[valid, 0], xy[valid, 1]
        cx, cy, R, rms = fit_circle_pivot_above(x, y, run.height)
        theta = np.full(len(t), np.nan)
        theta[valid] = np.arctan2(xy[valid, 0] - cx, xy[valid, 1] - cy)
        ys, noise, valid2 = smooth_and_noise(t, np.nan_to_num(theta), valid)
        # remove the equilibrium offset (camera roll): mean over whole periods
        offset = oscillation_center(t[valid2], theta[valid2])
        theta = theta - offset
        ys = ys - offset
        length_m = R * mpp if mpp > 0 else None
        cal = Calibration(meters_per_pixel=mpp, meters_per_pixel_std=mpp_std, method=method,
                          homography=homography, pivot_px=(cx, cy), length_px=R, length_m=length_m,
                          circle_fit_rms_px=rms, gravity_axis=(0.0, 1.0), theta_offset=offset,
                          object_radius_px=_object_radius(obj))
        ts = TimeSeries(name="theta", unit="rad", t=t.tolist(), y=np.nan_to_num(theta).tolist(),
                        y_smooth=ys.tolist(), noise_sigma=noise, valid_mask=valid2.tolist())
        return Measurement(cal, {"theta": ts}, "theta")

    if system_type == "falling_drag":
        g_axis = np.array([0.0, 1.0])
        pos = np.full(len(t), np.nan)
        pos[valid] = (xy[valid] @ g_axis) * mpp
        valid = trim_stationary(t, np.nan_to_num(pos), valid)
        ys, noise, valid2 = smooth_and_noise(t, np.nan_to_num(pos), valid)
        cal = Calibration(meters_per_pixel=mpp, meters_per_pixel_std=mpp_std, method=method,
                          homography=homography, gravity_axis=(0.0, 1.0), object_radius_px=_object_radius(obj))
        ts = TimeSeries(name="y", unit="m", t=t.tolist(), y=np.nan_to_num(pos).tolist(),
                        y_smooth=ys.tolist(), noise_sigma=noise, valid_mask=valid2.tolist())
        return Measurement(cal, {"y": ts}, "y")

    raise ValueError(f"unsupported system type {system_type}")


def series_arrays(ts: TimeSeries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.asarray(ts.t), np.asarray(ts.y), np.asarray(ts.valid_mask, bool)


PRIMARY_SERIES = {"pendulum": "theta", "falling_drag": "y"}
