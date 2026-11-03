"""Validity checks: Ghostline refuses to score data it can't trust (SPEC §7.5)."""
from __future__ import annotations

import numpy as np

from .config import get_settings
from .measure import centroids, reference_sizes_px, robust_std
from .schemas import Run, ValidityCheck, ValidityReport

_RANK = {"pass": 0, "warning": 1, "invalid": 2}


def _rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(x)))
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def check_camera(run: Run) -> list[ValidityCheck]:
    s = get_settings()
    ref = run.scene.reference_object
    tr = run.tracks.get(ref.concept) if run.tracks else None
    if tr is None:
        return [ValidityCheck(name="camera_moved", status="invalid",
                              reason=f"The reference object ({ref.concept}) was never tracked, so camera motion can't be checked.")]
    xy, valid = centroids(tr)
    t = np.asarray(run.frame_times[: len(xy)])
    out = []
    if valid.sum() < 5:
        return [ValidityCheck(name="camera_moved", status="invalid",
                              reason=f"The reference object ({ref.concept}) is visible in only {int(valid.sum())} frames.")]
    c, tv = xy[valid], t[valid]
    n0 = max(3, len(c) // 10)
    base = np.median(c[:n0], axis=0)
    sm = np.column_stack([_rolling_median(c[:, 0], 5), _rolling_median(c[:, 1], 5)])
    drift = np.hypot(*(sm - base).T)
    width = max(run.width, 1)
    dmax, i_d = float(drift.max()), int(drift.argmax())
    if dmax > s.camera_drift_frac * width:
        out.append(ValidityCheck(name="camera_drift", status="invalid", value=dmax / width, threshold=s.camera_drift_frac,
                                 reason=f"The reference object drifted {dmax:.0f} px ({100 * dmax / width:.1f}% of frame width) by t={tv[i_d]:.2f}s: the camera moved."))
    else:
        out.append(ValidityCheck(name="camera_drift", status="pass", value=dmax / width, threshold=s.camera_drift_frac,
                                 reason=f"Reference object stayed within {dmax:.1f} px of its starting position."))
    d = np.diff(c, axis=0)
    sig = max(robust_std(d[:, 0]), robust_std(d[:, 1]), 0.05)
    jumps = np.hypot(d[:, 0], d[:, 1])
    j, i_j = float(jumps.max()), int(jumps.argmax())
    thr = max(s.camera_jump_sigma * sig, 2.0)
    if j > thr:
        out.append(ValidityCheck(name="camera_jump", status="invalid", value=j / sig, threshold=s.camera_jump_sigma,
                                 reason=f"The reference object jumped {j:.1f} px between t={tv[i_j]:.2f}s and t={tv[i_j + 1]:.2f}s ({j / sig:.0f}σ of normal jitter): the camera was bumped."))
    else:
        out.append(ValidityCheck(name="camera_jump", status="pass", value=j / sig, threshold=s.camera_jump_sigma,
                                 reason="No sudden camera jumps."))
    sizes = reference_sizes_px(tr, ref)[valid]
    sm_s = _rolling_median(sizes, 15)
    med = float(np.median(sizes))
    change = float((sm_s.max() - sm_s.min()) / med) if med > 0 else 0.0
    if change > s.scale_change_frac:
        out.append(ValidityCheck(name="scale_changed", status="invalid", value=change, threshold=s.scale_change_frac,
                                 reason=f"The reference object's apparent size changed by {100 * change:.1f}%: the camera zoomed or the object moved toward the camera, so the scale isn't constant."))
    else:
        out.append(ValidityCheck(name="scale_changed", status="pass", value=change, threshold=s.scale_change_frac,
                                 reason=f"Scale stable (reference size varied {100 * change:.1f}%)."))
    return out


def check_tracking(run: Run, eval_start_t: float) -> ValidityCheck:
    s = get_settings()
    obj = run.tracks[run.scene.tracked_concepts[0]]
    _, valid = centroids(obj)
    ser = next(iter(run.series.values())) if run.series else None
    if ser is not None:
        valid = valid & np.asarray(ser.valid_mask, bool)[: len(valid)]
    t = np.asarray(run.frame_times[: len(valid)])
    if not valid.any():
        return ValidityCheck(name="tracking_lost", status="invalid", value=1.0, threshold=s.tracking_lost_invalid,
                             reason=f"The tracked object ({obj.concept}) was never found.")
    last = t[valid].max()
    win = (t > eval_start_t) & (t <= last)
    if not win.any():
        return ValidityCheck(name="tracking_lost", status="invalid", value=1.0, threshold=s.tracking_lost_invalid,
                             reason="No tracked frames in the evaluation window.")
    lost = float(1 - valid[win].mean())
    if lost > s.tracking_lost_invalid:
        st, why = "invalid", f"Tracking was lost in {100 * lost:.0f}% of the evaluation frames (limit {100 * s.tracking_lost_invalid:.0f}%)."
    elif lost > s.tracking_lost_warning:
        st, why = "warning", f"Tracking was lost in {100 * lost:.0f}% of the evaluation frames."
    else:
        st, why = "pass", f"Object tracked in {100 * (1 - lost):.1f}% of the evaluation frames."
    return ValidityCheck(name="tracking_lost", status=st, value=lost, threshold=s.tracking_lost_invalid, reason=why)


def check_timing(run: Run) -> ValidityCheck:
    s = get_settings()
    t = np.asarray(run.frame_times)
    if len(t) < 3:
        return ValidityCheck(name="frame_timing", status="invalid", reason="Too few frames.")
    d = np.diff(t)
    med = float(np.median(d))
    gaps = int((d > s.timing_gap_factor * med).sum())
    if gaps:
        return ValidityCheck(name="frame_timing", status="warning", value=float(d.max() / med), threshold=s.timing_gap_factor,
                             reason=f"{gaps} frame-timing gap(s) longer than {s.timing_gap_factor:.0f}x the median interval (dropped frames); real timestamps are used.")
    return ValidityCheck(name="frame_timing", status="pass", value=float(d.max() / med), threshold=s.timing_gap_factor,
                         reason=f"Frame timing regular (median {1 / med:.1f} fps).")


def check_pendulum_geometry(run: Run) -> ValidityCheck:
    cal = run.calibration
    rms, R = cal.circle_fit_rms_px or 0.0, cal.length_px or 1.0
    thr = max(2.0, 0.02 * R)
    if rms > thr:
        return ValidityCheck(name="pendulum_geometry", status="warning", value=rms, threshold=thr,
                             reason=f"The bob's path is not a clean arc (circle-fit residual {rms:.1f} px): string may stretch or the swing isn't planar.")
    return ValidityCheck(name="pendulum_geometry", status="pass", value=rms, threshold=thr,
                         reason=f"Bob moves on a circular arc (residual {rms:.2f} px, length {cal.length_m:.3f} m).")


def check_setup(run: Run, reference: dict) -> list[ValidityCheck]:
    s = get_settings()
    out = []
    L0, L = reference.get("length_m"), run.calibration.length_m if run.calibration else None
    if L0 and L:
        diff = abs(L - L0) / L0
        st = "invalid" if diff > s.setup_mismatch_frac else "pass"
        why = (f"Pendulum length {L:.3f} m differs {100 * diff:.1f}% from the committed setup ({L0:.3f} m): this looks like a different setup."
               if st == "invalid" else f"Same pendulum as the committed experiment (length {L:.3f} m vs {L0:.3f} m).")
        out.append(ValidityCheck(name="setup_mismatch", status=st, value=diff, threshold=s.setup_mismatch_frac, reason=why))
    m0, m = reference.get("meters_per_pixel"), run.calibration.meters_per_pixel if run.calibration else None
    if m0 and m and reference.get("system_type") == "falling_drag":
        diff = abs(m - m0) / m0
        st = "warning" if diff > s.setup_mismatch_frac else "pass"
        out.append(ValidityCheck(name="camera_distance", status=st, value=diff, threshold=s.setup_mismatch_frac,
                                 reason=f"Image scale differs {100 * diff:.1f}% from the committed run." if st == "warning"
                                 else "Camera distance matches the committed run."))
    return out


def check_validity(run: Run, system_type: str, reference: dict | None = None) -> ValidityReport:
    checks: list[ValidityCheck] = []
    checks += check_camera(run)
    if run.series:
        ser = next(iter(run.series.values()))
        valid = np.asarray(ser.valid_mask, bool)
        t = np.asarray(ser.t)
        t_first = float(t[valid].min()) if valid.any() else 0.0
        if run.role == "reveal":
            start = t_first + (run.split.conditioning_window_s if run.split else get_settings().conditioning_window_s)
        else:
            start = run.split.train_end_t if run.split else t_first
        checks.append(check_tracking(run, start))
    checks.append(check_timing(run))
    if system_type == "pendulum" and run.calibration and run.calibration.length_px:
        checks.append(check_pendulum_geometry(run))
    if reference:
        checks += check_setup(run, reference)
    worst = max((_RANK[c.status] for c in checks), default=0)
    status = {0: "valid", 1: "warning", 2: "invalid"}[worst]
    return ValidityReport(status=status, checks=checks)
