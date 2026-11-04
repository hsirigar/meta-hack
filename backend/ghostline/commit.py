"""Commit predictions before the experiment is filmed; score the reveal (SPEC §12)."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.optimize import least_squares

from .fit import physics_of
from .models.base import get_model
from .models.pendulum import PERIOD_WINDOW_S, measure_period
from .schemas import Commitment, CommittedModel, FitResult, ValidityReport, Verdict

SIG_DIGITS = 9
ADEQUATE_SIGMA = 20.0     # winner RMSE above this many noise sigmas ...
ADEQUATE_OBS_PCT = 1.0    # ... and observable error above this percent => no candidate describes the data


class CommitmentTampered(RuntimeError):
    pass


# ---------- canonical JSON ----------
def _canon(obj):
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("non-finite number in commitment")
        r = float(f"{obj:.{SIG_DIGITS}g}")
        return int(r) if r.is_integer() and abs(r) < 1e15 else r
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, np.generic):
        return _canon(obj.item())
    raise TypeError(f"cannot canonicalise {type(obj)}")


def canonical_json(content: dict) -> str:
    """Sorted keys, floats rounded to 9 significant digits, no whitespace."""
    return json.dumps(_canon(content), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


PROTOCOL = {
    "conditioning": "initial state (position, velocity) estimated from the first conditioning_window_s seconds of valid data only; physical parameters frozen",
    "metric": "RMSE of the observed variable after the conditioning window",
    "observable": "pendulum: mean period over the 10 s after the conditioning window; drag: late-fall speed",
    "success_rule": "winner = lowest RMSE after the conditioning window, provided the run is valid",
}


def build_commitment(experiment_id: str, fits: list[FitResult], knob: str, ref_knob: float, value: float,
                     created_at: str, conditioning_window_s: float, reference: dict,
                     noise_sigma: float, source_run_id: str) -> Commitment:
    models = []
    for f in fits:
        m = get_model(f.model)
        phys = physics_of(f)
        phys_new = m.transfer(phys, ref_knob, value)
        models.append(CommittedModel(
            name=f.model, params=phys, param_std={k: f.param_std[k] for k in phys},
            knob_transfer=m.knob_transfer,
            predicted_observables={**m.predicted_observables(phys_new, value),
                                   **{f"{k}_at_target": v for k, v in phys_new.items()}}))
    content = {
        "experiment_id": experiment_id,
        "created_at": created_at,
        "knob_setting": {knob: value},
        "reference_knob": {knob: ref_knob},
        "source_run_id": source_run_id,
        "models": [cm.model_dump() for cm in models],
        "protocol": {**PROTOCOL, "conditioning_window_s": conditioning_window_s, "noise_sigma_ref": noise_sigma},
        "reference": reference,
    }
    cj = canonical_json(content)
    sha = sha256_hex(cj)
    return Commitment(id=f"c_{sha[:12]}", experiment_id=experiment_id, created_at=created_at,
                      knob_setting={knob: value}, models=models, protocol=content["protocol"],
                      reference=reference, canonical_json=cj, sha256=sha)


def verify_commitment(c: Commitment) -> bool:
    """Hash matches, canonical JSON is canonical, and the structured copy agrees with the hashed content."""
    if sha256_hex(c.canonical_json) != c.sha256:
        return False
    try:
        content = json.loads(c.canonical_json)
    except json.JSONDecodeError:
        return False
    if canonical_json(content) != c.canonical_json:
        return False
    structured = canonical_json({"models": [m.model_dump() for m in c.models], "knob_setting": c.knob_setting,
                                 "experiment_id": c.experiment_id})
    hashed = canonical_json({"models": content["models"], "knob_setting": content["knob_setting"],
                             "experiment_id": content["experiment_id"]})
    return structured == hashed


def committed_content(c: Commitment) -> dict:
    if not verify_commitment(c):
        raise CommitmentTampered(f"commitment {c.id} failed hash verification")
    return json.loads(c.canonical_json)


# ---------- reveal ----------
def condition_initial_state(model_name: str, phys: dict, t: np.ndarray, y: np.ndarray, noise_sigma: float):
    """Estimate only the initial state at t[0] from the conditioning window (params frozen)."""
    m = get_model(model_name)
    t0 = float(t[0])
    lo = np.array([p.lower for p in m.state_params]); hi = np.array([p.upper for p in m.state_params])
    guesses = m.initial_guesses(t, y)
    n_phys = len(m.physics_params)

    def res(x0):
        return m.simulate(t, phys, x0, t0) - y

    best = None
    for g in guesses[:3]:
        x_start = np.clip(np.asarray(g[n_phys:], float), lo + 1e-9, hi - 1e-9)
        sol = least_squares(res, x_start, bounds=(lo, hi), method="trf", loss="soft_l1",
                            f_scale=max(3 * noise_sigma, 1e-9), x_scale="jac", diff_step=1e-6)
        if best is None or sol.cost < best.cost:
            best = sol
    return [float(v) for v in best.x], t0


def _observed_observable(system: str, t: np.ndarray, y: np.ndarray) -> float | None:
    if system == "pendulum":
        w = t <= t[0] + PERIOD_WINDOW_S if len(t) else t
        return measure_period(t[w], y[w])
    if len(t) >= 6:  # late-fall speed: slope over the last 30% of the evaluated window
        k = max(int(len(t) * 0.3), 4)
        return float(np.polyfit(t[-k:], y[-k:], 1)[0])
    return None


def evaluate_commitment(c: Commitment, run_id: str, t: np.ndarray, y: np.ndarray, valid: np.ndarray,
                        noise_sigma: float, validity: ValidityReport) -> tuple[Verdict, dict]:
    """Score a reveal run against a commitment. Returns (Verdict, per-model prediction curves)."""
    hash_ok = verify_commitment(c)
    if not hash_ok:
        return Verdict(commitment_id=c.id, run_id=run_id, hash_verified=False, validity=validity, per_model=[],
                       winner=None, summary_facts={"refused": "commitment hash mismatch: predictions were altered"}), {}
    if validity.status == "invalid":
        return Verdict(commitment_id=c.id, run_id=run_id, hash_verified=True, validity=validity, per_model=[],
                       winner=None, summary_facts={"refused": "run is INVALID; commitment not scored",
                                                   "reasons": validity.reasons}), {}
    content = committed_content(c)
    window = float(content["protocol"]["conditioning_window_s"])
    knob, value = next(iter(content["knob_setting"].items()))
    ref_knob = content["reference_knob"][knob]
    tv, yv = np.asarray(t)[valid], np.asarray(y)[valid]
    t_start = float(tv[0])
    cond = tv <= t_start + window
    ev = ~cond
    per_model, curves = [], {"t": np.asarray(t).tolist()}
    system = get_model(content["models"][0]["name"]).system
    obs_measured = _observed_observable(system, tv[ev], yv[ev])
    for cm in content["models"]:
        m = get_model(cm["name"])
        phys = m.transfer(cm["params"], ref_knob, value)
        x0, t0 = condition_initial_state(cm["name"], phys, tv[cond], yv[cond], noise_sigma)
        pred_ev = m.simulate(tv[ev], phys, x0, t0)
        rmse = float(np.sqrt(np.mean((pred_ev - yv[ev]) ** 2)))
        obs_pred = _observed_observable(system, tv[ev], pred_ev)
        entry = {"model": cm["name"], "rmse": rmse, "rmse_sigma": rmse / max(noise_sigma, 1e-12),
                 "conditioned_state": x0, "params_used": phys,
                 "committed_observables": cm["predicted_observables"],
                 "observable": m.observable, "observable_predicted": obs_pred, "observable_measured": obs_measured}
        if obs_pred is not None and obs_measured is not None:
            entry["observable_error"] = obs_pred - obs_measured
            entry["observable_error_pct"] = 100 * (obs_pred - obs_measured) / obs_measured if obs_measured else None
        per_model.append(entry)
        curves[cm["name"]] = m.simulate(np.asarray(t, float), phys, x0, t0).tolist()
    per_model.sort(key=lambda e: e["rmse"])
    winner = per_model[0]["model"] if per_model else None
    facts = {"knob": knob, "value": value, "conditioning_window_s": window, "noise_sigma": noise_sigma,
             "eval_samples": int(ev.sum()), "eval_start_t": t_start + window, "observable_measured": obs_measured,
             "rmse_ratio_loser_over_winner": (per_model[-1]["rmse"] / max(per_model[0]["rmse"], 1e-12)) if len(per_model) > 1 else None}
    if per_model:
        w = per_model[0]
        err_pct = abs(w.get("observable_error_pct") or 0.0)
        # Ranking says which candidate is least wrong; adequacy says whether any of them actually describes the data.
        facts["winner_rmse_sigma"] = w["rmse_sigma"]
        facts["winner_observable_error_pct"] = err_pct
        facts["all_models_inadequate"] = bool(w["rmse_sigma"] > ADEQUATE_SIGMA and err_pct > ADEQUATE_OBS_PCT)
        if facts["all_models_inadequate"]:
            facts["inadequacy_note"] = (f"Even the best model misses by {w['rmse_sigma']:.0f}σ and its {w['observable']} is off by "
                                        f"{err_pct:.1f}%: the system is outside the candidate set.")
            shape = next((c for c in validity.checks if c.name == "object_shape" and c.status == "warning"), None)
            if shape is not None:
                facts["likely_cause"] = (f"the tracked object is not a point mass: its outline changes size by {100 * shape.value:.0f}% "
                                         "as it twists and tilts during the swing")
                facts["inadequacy_note"] += f" Likely cause: {facts['likely_cause']}."
    return Verdict(commitment_id=c.id, run_id=run_id, hash_verified=True, validity=validity, per_model=per_model,
                   winner=winner, summary_facts=facts), curves
