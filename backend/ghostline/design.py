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


DECISIVE_SIGMA = 10.0


def relative_effect(fits: list[FitResult], value: float, ref_knob: float) -> float:
    """Largest pairwise relative difference in the models' predicted observable at this setting."""
    obs = []
    for f in fits:
        m = get_model(f.model)
        p = m.transfer(physics_of(f), ref_knob, value)
        obs.append(m.predicted_observables(p, value)[m.observable])
    best = 0.0
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            best = max(best, abs(obs[i] - obs[j]) / max(abs(obs[i] + obs[j]) / 2, 1e-12))
    return best


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
               fps: float = 60.0, window_s: float = 0.5) -> tuple[float, float, dict]:
    """Expected separation under the commit/reveal protocol.

    For each ordered pair (A = truth, B = candidate): simulate what the camera would see if A were true,
    condition B's initial state on the first `window_s` of that footage (params frozen, exactly as in a
    reveal), then measure how far B's prediction drifts from A afterwards. Returns (rms σ, max σ, curves).
    """
    from .commit import condition_initial_state

    models = [get_model(f.model) for f in fits]
    phys = [m.transfer(physics_of(f), ref_knob, value) for m, f in zip(models, fits)]
    horizon = max(m.design_horizon(p, value) for m, p in zip(models, phys))
    t = np.arange(0, horizon, 1 / fps)
    s = max(noise_sigma, 1e-12)
    rms, mx, curves = 0.0, 0.0, {"t": t}
    for a, (fa, ma, pa) in enumerate(zip(fits, models, phys)):
        ya = ma.simulate(t, pa, ma.default_initial_state(value), 0.0)
        obs = ma.observed_mask(t, ya)
        to, yo = t[obs], ya[obs]
        if len(to) < 10:
            continue
        cond = to <= to[0] + window_s
        curves[fa.model] = ya
        for b, (fb, mb, pb) in enumerate(zip(fits, models, phys)):
            if a == b or cond.all():
                continue
            x0, t0 = condition_initial_state(fb.model, pb, to[cond], yo[cond], s)
            d = mb.simulate(to[~cond], pb, x0, t0) - yo[~cond]
            rms = max(rms, float(np.sqrt(np.mean(d ** 2))))
            mx = max(mx, float(np.max(np.abs(d))))
    return rms / s, mx / s, curves


def design_experiment(fits: list[FitResult], knob: str, ref_knob: float, noise_sigma: float,
                      source_run_id: str = "", max_value: float | None = None) -> ExperimentProposal:
    grid = [v for v in KNOB_GRIDS[knob] if max_value is None or v <= max_value + 1e-9]
    if not grid:
        grid = [min(KNOB_GRIDS[knob])]
    rows = []
    for v in grid:
        rms, mx, _ = separation(fits, knob, v, ref_knob, noise_sigma)
        feas, _ = feasibility(knob, v)
        rows.append({"value": v, "rms": rms, "max": mx, "feas": feas,
                     "effect": relative_effect(fits, v, ref_knob)})
    curve = [(r["value"], r["rms"]) for r in rows]
    # 1) settings that are decisive with margin and safe to perform;
    # 2) among them, the largest effect in the physical observable (robust to unmodelled systematics).
    ok = [r for r in rows if r["rms"] >= DECISIVE_SIGMA and r["feas"] >= 0.9]
    if ok:
        best = max(ok, key=lambda r: r["effect"])
        why = (f"it is the feasible setting with the largest physical effect ({100 * best['effect']:.0f}% difference in the "
               f"predicted observable) among settings that separate the models by ≥{DECISIVE_SIGMA:g}σ")
    else:
        best = max(rows, key=lambda r: np.log1p(r["rms"]) * r["feas"])
        why = "no feasible setting reaches a decisive margin; this maximises separation with a feasibility penalty"
    v, rms, mx = best["value"], best["rms"], best["max"]
    _, note = feasibility(knob, v)
    unit = "°" if knob == "release_angle_deg" else " filters"
    cur = dict(curve).get(ref_knob)
    cur_txt = f"{cur:.1f}σ at the current setting ({ref_knob:g}{unit})" if cur is not None else ""
    limit_txt = (f"Searched only up to {max_value:g}{unit}, the most this setup allows. " if max_value is not None else "")
    feas_note = (f"Recommend {v:g}{unit} because {why}. Expected separation {rms:.1f}σ RMS (max {mx:.0f}σ) after the "
                 f"0.5 s conditioning window. " + (f"Compare {cur_txt}. " if cur_txt else "") + limit_txt +
                 ("" if max_value is not None else
                  (f"Larger values were avoided because {feasibility(knob, grid[-1])[1]}." if note == "" else f"Caution: {note}.")))
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
