"""Experiment design: which knob setting best separates the candidate models? (SPEC §11)"""
from __future__ import annotations

import numpy as np

from .fit import physics_of
from .models.base import get_model
from .schemas import ExperimentProposal, FitResult

KNOB_GRIDS = {
    "release_angle_deg": [float(a) for a in range(5, 81, 5)],
    "n_filters": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
DEFAULT_KNOB = {"pendulum": "release_angle_deg", "falling_drag": "n_filters"}


def feasibility(knob: str, value: float) -> tuple[float, str]:
    if knob == "release_angle_deg":
        if value <= 70:
            return 1.0, ""
        f = max(0.0, 1 - ((value - 70) / 12) ** 2)
        return f, "above ~75° the string can go slack and the bob may leave the frame"
    if knob == "n_filters":
        return {5.0: 0.5, 6.0: 0.15}.get(value, 1.0), ("stacks above 5 filters tend to tumble" if value >= 5 else "")
    return 1.0, ""


def separation(fits: list[FitResult], knob: str, value: float, ref_knob: float, noise_sigma: float,
               fps: float = 60.0) -> tuple[float, float, dict]:
    """Simulate every model at `value` from the same initial state; return (rms sep σ, max sep σ, curves)."""
    models = [get_model(f.model) for f in fits]
    phys = [m.transfer(physics_of(f), ref_knob, value) for m, f in zip(models, fits)]
    horizon = max(m.design_horizon(p, value) for m, p in zip(models, phys))
    t = np.arange(0, horizon, 1 / fps)
    preds = {f.model: m.simulate(t, p, m.default_initial_state(value), 0.0) for f, m, p in zip(fits, models, phys)}
    keys = list(preds)
    rms, mx = 0.0, 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = preds[keys[i]] - preds[keys[j]]
            rms = max(rms, float(np.sqrt(np.mean(d ** 2))))
            mx = max(mx, float(np.max(np.abs(d))))
    s = max(noise_sigma, 1e-12)
    return rms / s, mx / s, {"t": t, **preds}


def design_experiment(fits: list[FitResult], knob: str, ref_knob: float, noise_sigma: float,
                      source_run_id: str = "") -> ExperimentProposal:
    grid = KNOB_GRIDS[knob]
    curve, best, best_u = [], None, -1.0
    for v in grid:
        rms, mx, _ = separation(fits, knob, v, ref_knob, noise_sigma)
        feas, _ = feasibility(knob, v)
        u = np.log1p(rms) * feas  # diminishing returns on separation, penalise risky setups
        curve.append((v, rms))
        if u > best_u:
            best, best_u = (v, rms, mx), u
    v, rms, mx = best
    _, note = feasibility(knob, v)
    unit = "°" if knob == "release_angle_deg" else " filters"
    cur = dict(curve).get(ref_knob)
    cur_txt = f"{cur:.1f}σ at the current setting ({ref_knob:g}{unit})" if cur is not None else ""
    feas_note = (f"Best trade-off between separation and feasibility at {v:g}{unit}: models separate by {rms:.1f}σ "
                 f"(RMS; max {mx:.0f}σ). " + (f"Compare {cur_txt}. " if cur_txt else "") +
                 (f"Larger values were penalised because {feasibility(knob, grid[-1])[1]}." if note == "" else f"Caution: {note}."))
    if knob == "release_angle_deg":
        instr = (f"Release the same pendulum from {v:g}° (string taut, bob at rest), keep the camera and the reference "
                 f"object exactly where they were, and record at least 6 full swings.")
    else:
        instr = (f"Stack {int(v)} coffee filters (nested), drop them from the same height (~2 m) in front of the same wall, "
                 f"with the camera and reference card unchanged.")
    return ExperimentProposal(knob=knob, value=v, expected_separation_sigma=rms, expected_max_separation_sigma=mx,
                              discrimination_curve=curve, feasibility_note=feas_note,
                              models=[f.model for f in fits], source_run_id=source_run_id,
                              filming_instructions=instr)
