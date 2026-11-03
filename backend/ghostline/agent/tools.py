"""Agent tools: JSON-schema function definitions over deterministic Python (SPEC §13)."""
from __future__ import annotations

import math
from typing import Any, Callable

from .. import pipeline as P
from .. import storage
from ..render import LABELS
from ..scene import scene_backend_label


def _r(x, n=4):
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return float(f"{x:.{n}g}")
    if isinstance(x, dict):
        return {k: _r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    return x


def _fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}


S = {"type": "string"}
SL = {"type": "array", "items": {"type": "string"}}

TOOL_DEFS = [
    _fn("describe_scene", "Understand the scene from keyframes: system type, what to track, reference object of known size.", {"run_id": S}, ["run_id"]),
    _fn("track_objects", "Track the moving object and the reference object through the video (SAM 3.1 or fallback tracker).", {"run_id": S, "concepts": SL}, ["run_id"]),
    _fn("measure", "Convert tracks into calibrated physical time series (theta(t) or y(t)), estimate noise, fix the train/holdout split.", {"run_id": S}, ["run_id"]),
    _fn("check_validity", "Check whether the recording can be trusted (camera moved, scale changed, tracking lost, timing, setup mismatch).", {"run_id": S}, ["run_id"]),
    _fn("list_models", "List candidate dynamical models for a system type.", {"system_type": {"type": "string", "enum": ["pendulum", "falling_drag"]}}, ["system_type"]),
    _fn("fit_models", "Fit candidate models on the TRAINING part of the run only, then score them on the held-out part.", {"run_id": S, "model_names": SL}, ["run_id", "model_names"]),
    _fn("compare_models", "Compare fitted models on holdout evidence; returns decisive/ambiguous with the numbers behind it.", {"run_id": S}, ["run_id"]),
    _fn("discover_equation", "Sparse equation discovery (PySINDy) on the training data, as supporting evidence.", {"run_id": S}, ["run_id"]),
    _fn("design_experiment", "Find the experimental setting (knob value) that best separates the candidate models.",
        {"experiment_id": S, "model_names": SL, "knob": {"type": "string", "enum": ["release_angle_deg", "n_filters"]}}, ["experiment_id"]),
    _fn("commit_predictions", "Freeze each model's predictions for the proposed experiment and publish a SHA-256 commitment BEFORE the user films it.",
        {"experiment_id": S, "model_names": SL, "knob_setting": {"type": "object", "additionalProperties": {"type": "number"}}}, ["experiment_id"]),
    _fn("evaluate_commitment", "Score a new (reveal) run against a commitment: verify hash, check validity, condition on the first 0.5 s only, compare predictions.",
        {"commitment_id": S, "run_id": S}, ["commitment_id", "run_id"]),
    _fn("render_ghosts", "Render each model's prediction as a translucent ghost over the video.", {"run_id": S, "model_names": SL}, ["run_id"]),
]


# ---------- implementations (return compact digests) ----------
def t_describe_scene(run_id: str) -> dict:
    s = P.describe_scene(run_id)
    return {**s.model_dump(), "backend": scene_backend_label()}


def t_track_objects(run_id: str, concepts: list[str] | None = None) -> dict:
    return {"tracks": P.track_objects(run_id, concepts or None)}


def t_measure(run_id: str) -> dict:
    return _r(P.measure(run_id), 5)


def t_check_validity(run_id: str) -> dict:
    rep = P.check_validity(run_id)
    return {"status": rep.status, "checks": [{"name": c.name, "status": c.status, "reason": c.reason} for c in rep.checks]}


def t_list_models(system_type: str) -> dict:
    return {"models": [{"name": m["name"], "equation": m["latex"], "params": [p["name"] for p in m["params"]],
                        "knob_transfer": m["knob_transfer"]} for m in P.list_model_specs(system_type)]}


def t_fit_models(run_id: str, model_names: list[str]) -> dict:
    fits = P.fit_run(run_id, model_names)
    return {"note": "fitted on training window only; holdout scored with frozen parameters",
            "fits": [_r({"model": f.model, "params": {k: v for k, v in f.params.items() if k != "t_ref"},
                         "param_std": {k: v for k, v in f.param_std.items() if k != "t_ref"},
                         "train_rmse": f.train_rmse, "holdout_rmse": f.holdout_rmse, "bic": f.bic,
                         "ljung_box_p": f.residual_whiteness_p, "n_train": f.n_train, "n_holdout": f.n_holdout}) for f in fits]}


def t_compare_models(run_id: str) -> dict:
    c = P.compare_run(run_id)
    f = c.explanation_facts
    return _r({"status": c.status, "best": c.best, "max_separation_sigma": c.max_separation_sigma,
               "delta_bic": f.get("delta_bic"), "holdout_rmse_ratio": f.get("holdout_rmse_ratio"),
               "noise_sigma": f.get("noise_sigma"), "rule": f.get("rule"),
               "holdout_rmse": {k: v["holdout_rmse"] for k, v in f["models"].items()},
               "holdout_rmse_in_noise_sigma": {k: v["holdout_rmse_sigma"] for k, v in f["models"].items()},
               "implied_length_m": {k: v.get("implied_length_m") for k, v in f["models"].items()},
               "measured_length_m": f.get("measured_length_m"), "amplitude": f.get("amplitude")})


def t_discover_equation(run_id: str) -> dict:
    return _r(P.discover(run_id), 4)


def t_design_experiment(experiment_id: str, model_names: list[str] | None = None, knob: str | None = None) -> dict:
    p = P.design(experiment_id, model_names or None, knob)
    return _r({"knob": p.knob, "recommended_value": p.value, "expected_separation_sigma": p.expected_separation_sigma,
               "expected_max_separation_sigma": p.expected_max_separation_sigma,
               "discrimination_curve": [{"value": a, "separation_sigma": b} for a, b in p.discrimination_curve],
               "feasibility_note": p.feasibility_note, "filming_instructions": p.filming_instructions}, 4)


def t_commit_predictions(experiment_id: str, model_names: list[str] | None = None, knob_setting: dict | None = None) -> dict:
    c = P.commit(experiment_id, model_names or None, knob_setting or None)
    return _r({"commitment_id": c.id, "sha256": c.sha256, "hash12": c.sha256[:12], "created_at": c.created_at,
               "knob_setting": c.knob_setting,
               "predictions": {m.name: m.predicted_observables for m in c.models},
               "protocol": c.protocol["success_rule"]}, 5)


def t_evaluate_commitment(commitment_id: str, run_id: str) -> dict:
    v = P.evaluate(commitment_id, run_id)
    out = {"hash_verified": v.hash_verified, "validity": v.validity.status, "winner": v.winner,
           "summary": v.summary_facts,
           "per_model": [{k: e.get(k) for k in ("model", "rmse", "rmse_sigma", "observable", "observable_predicted",
                                                "observable_measured", "observable_error_pct")} for e in v.per_model]}
    if v.validity.status == "invalid":
        out["validity_reasons"] = v.validity.reasons
    return _r(out, 4)


def t_render_ghosts(run_id: str, model_names: list[str] | None = None) -> dict:
    path = P.render(run_id, model_names or None)
    r = storage.load_run(run_id)
    from ..pipeline import load_curves
    shown = [k for k in load_curves(run_id) if k != "t"] if not (r.validity and r.validity.status == "invalid") else []
    return {"overlay_url": f"/api/runs/{run_id}/overlay.mp4", "ghosts": [LABELS.get(k, k) for k in shown], "path": path}


IMPL: dict[str, Callable[..., dict]] = {
    "describe_scene": t_describe_scene, "track_objects": t_track_objects, "measure": t_measure,
    "check_validity": t_check_validity, "list_models": t_list_models, "fit_models": t_fit_models,
    "compare_models": t_compare_models, "discover_equation": t_discover_equation,
    "design_experiment": t_design_experiment, "commit_predictions": t_commit_predictions,
    "evaluate_commitment": t_evaluate_commitment, "render_ghosts": t_render_ghosts,
}


def execute(name: str, args: dict[str, Any]) -> dict:
    if name not in IMPL:
        return {"error": f"unknown tool {name}"}
    try:
        return IMPL[name](**args)
    except Exception as e:  # tool errors go back to the agent as data
        return {"error": f"{type(e).__name__}: {e}"}
