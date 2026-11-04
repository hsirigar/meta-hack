"""Orchestration over storage. Every agent tool and API route goes through here."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import storage
from .commit import build_commitment, evaluate_commitment
from .compare import compare_fits
from .config import get_settings
from .design import DEFAULT_KNOB, design_experiment
from .fit import bootstrap_band, fit_models, predict, score_holdout
from .ingest import ingest
from .measure import PRIMARY_SERIES, measure_run, series_arrays
from .models.base import DEFAULT_CANDIDATES, models_for
from .scene import describe_scene as scene_describe
from .schemas import (Commitment, Comparison, Experiment, ExperimentProposal, FitResult, Run,
                      SceneDescription, Split, ValidityReport, Verdict)
from .segment import track_run, track_summary
from .validity import check_validity as run_validity


# ---------- experiments / runs ----------
def create_experiment(name: str, system_type: str, description: str = "", exp_id: str | None = None,
                      knob_max: float | None = None) -> Experiment:
    e = Experiment(id=exp_id or storage.new_id("exp"), name=name, system_type=system_type,
                   description=description, knob_max=knob_max, created_at=storage.now_iso())
    return storage.save_experiment(e)


def add_run(experiment_id: str, video: Path, knobs: dict[str, float] | None = None, role: str = "exploration",
            commitment_id: str | None = None, run_id: str | None = None) -> Run:
    e = storage.load_experiment(experiment_id)
    rid = run_id or storage.new_id("run")
    info = ingest(Path(video), rid)
    r = Run(id=rid, experiment_id=experiment_id, knobs=knobs or {}, role=role, commitment_id=commitment_id,
            created_at=storage.now_iso(), **info)
    storage.save_run(r)
    e.run_ids.append(rid)
    storage.save_experiment(e)
    return r


def _exp_runs(e: Experiment, role: str | None = None) -> list[Run]:
    runs = [storage.load_run(r) for r in e.run_ids]
    return [r for r in runs if role is None or r.role == role]


def source_run(e: Experiment) -> Run | None:
    """Latest exploration run that has fits."""
    runs = [r for r in _exp_runs(e, "exploration") if r.fits]
    return runs[-1] if runs else None


# ---------- scene / tracking / measurement ----------
def describe_scene(run_id: str) -> SceneDescription:
    r = storage.load_run(run_id)
    e = storage.load_experiment(r.experiment_id)
    prior = [x for x in _exp_runs(e) if x.scene is not None and x.id != r.id]
    if r.role == "reveal" and prior:
        scene = prior[0].scene.model_copy(deep=True)  # same setup: reuse prompts and reference
        scene.notes = (scene.notes + " | reused from committed setup").strip(" |")
    else:
        scene = scene_describe([storage.abs_path(k) for k in r.keyframes], e.system_type, user_note=e.description)
    r.scene = scene
    storage.save_run(r)
    return scene


def set_scene(run_id: str, scene: SceneDescription) -> Run:
    r = storage.load_run(run_id)
    r.scene = scene
    return storage.save_run(r)


def track_objects(run_id: str, concepts: list[str] | None = None) -> dict:
    r = storage.load_run(run_id)
    if r.scene is None:
        describe_scene(run_id)
        r = storage.load_run(run_id)
    concepts = concepts or (r.scene.tracked_concepts + [r.scene.reference_object.concept])
    if r.scene.reference_object.concept not in concepts:
        concepts = concepts + [r.scene.reference_object.concept]
    r.tracks = track_run(r, concepts)
    storage.save_run(r)
    return track_summary(r.tracks)


def measure(run_id: str) -> dict:
    r = storage.load_run(run_id)
    e = storage.load_experiment(r.experiment_id)
    m = measure_run(r, e.system_type, homography=r.calibration.homography if r.calibration and r.calibration.homography else None)
    r.calibration, r.series = m.calibration, m.series
    ts = m.series[m.primary]
    t, y, valid = series_arrays(ts)
    if r.split is None:  # fixed once, never changes afterwards
        tv = t[valid]
        cw = get_settings().conditioning_window_s
        if r.role == "reveal":
            r.split = Split(train_end_t=float(tv[0] + cw), conditioning_window_s=cw)
        else:
            r.split = Split(train_end_t=float(tv[0] + get_settings().train_fraction * (tv[-1] - tv[0])),
                            conditioning_window_s=cw)
    if e.system_type == "pendulum" and "release_angle_deg" not in r.knobs:
        k = max(int(0.1 * valid.sum()), 5)
        r.knobs["release_angle_deg"] = round(float(np.degrees(np.max(np.abs(y[valid][:k * 3])))), 1)
    if e.system_type == "falling_drag" and "n_filters" not in r.knobs:
        r.knobs["n_filters"] = 1.0
    storage.save_run(r)
    tv = t[valid]
    out = {"series": ts.name, "unit": ts.unit, "n_frames": len(t), "n_valid": int(valid.sum()),
           "duration_s": float(tv[-1] - tv[0]) if len(tv) else 0.0, "noise_sigma": ts.noise_sigma,
           "meters_per_pixel": m.calibration.meters_per_pixel, "calibration_method": m.calibration.method,
           "split_train_end_t": r.split.train_end_t, "knobs": r.knobs}
    if m.calibration.length_m:
        out.update({"pendulum_length_m": m.calibration.length_m, "pivot_px": m.calibration.pivot_px,
                    "amplitude_deg": float(np.degrees(np.max(np.abs(y[valid]))))})
    if e.system_type == "falling_drag" and len(tv) > 4:
        out["mean_speed_mps"] = float((y[valid][-1] - y[valid][0]) / (tv[-1] - tv[0]))
    return out


def commitment_reference(r: Run, system_type: str) -> dict:
    ref = {"system_type": system_type, "meters_per_pixel": r.calibration.meters_per_pixel}
    if system_type == "pendulum" and not r.calibration.length_m and r.fits:
        # angles-only clip (no object of known size): the setup is identified by its swing timing
        from .models.base import G
        implied = [G / f.params["omega0"] ** 2 for f in r.fits if "omega0" in f.params]
        ref["length_m"], ref["length_source"] = float(np.mean(implied)), "implied by fitted omega0 (no metre scale in the video)"
    elif r.calibration.length_m:
        t, y, valid = series_arrays(r.series["theta"])
        span = float(np.degrees(np.ptp(y[valid])))
        ref["measured_length_m"] = r.calibration.length_m
        ref["arc_span_deg"] = span
        if span >= 20 or not r.fits:
            ref["length_m"], ref["length_source"] = r.calibration.length_m, "circle fit of the bob path"
        else:
            # short arcs make the circle fit ill-conditioned: use the pendulum's implied length g/omega0^2
            from .models.base import G
            implied = [G / f.params["omega0"] ** 2 for f in r.fits if "omega0" in f.params]
            ref["length_m"], ref["length_source"] = float(np.mean(implied)), "implied by fitted omega0 (short arc)"
    return ref


def check_validity(run_id: str) -> ValidityReport:
    r = storage.load_run(run_id)
    e = storage.load_experiment(r.experiment_id)
    reference = None
    if r.role == "reveal" and r.commitment_id:
        reference = storage.load_commitment(r.commitment_id).reference
    rep = run_validity(r, e.system_type, reference)
    r.validity = rep
    storage.save_run(r)
    return rep


# ---------- fitting / comparison ----------
def _arrays(r: Run, e: Experiment):
    ts = r.series[PRIMARY_SERIES[e.system_type]]
    return (*series_arrays(ts), ts.noise_sigma)


def fit_run(run_id: str, model_names: list[str] | None = None, bootstrap: int = 40) -> list[FitResult]:
    r = storage.load_run(run_id)
    e = storage.load_experiment(r.experiment_id)
    if r.role == "reveal":
        raise PermissionError("reveal runs are never refit: score them against the commitment instead")
    model_names = model_names or DEFAULT_CANDIDATES[e.system_type]
    t, y, valid, noise = _arrays(r, e)
    train = valid & (t <= r.split.train_end_t)
    hold = valid & (t > r.split.train_end_t)
    # LEAKAGE GUARD: fitting sees only the training slice.
    fits = fit_models(t[train].copy(), y[train].copy(), model_names, noise, run_id)
    fits = [score_holdout(f, t[hold], y[hold]) for f in fits]
    if bootstrap:
        grid = t[valid]
        fits = [f.model_copy(update={"band": bootstrap_band(f, t[train], y[train], grid, noise, n_boot=bootstrap)})
                for f in fits]
    r.fits = fits
    r.comparison = None
    storage.save_run(r)
    _save_curves(r, {f.model: predict(f, t) for f in fits})
    return fits


def _save_curves(r: Run, curves: dict[str, np.ndarray]) -> None:
    t = r.series[next(iter(r.series))].t
    storage.save_json(storage.run_dir(r.id) / "curves.json",
                      {"t": t, **{k: np.asarray(v).tolist() for k, v in curves.items()}})


def load_curves(run_id: str) -> dict:
    p = storage.run_dir(run_id) / "curves.json"
    return storage.load_json(p) if p.exists() else {}


def compare_run(run_id: str) -> Comparison:
    r = storage.load_run(run_id)
    e = storage.load_experiment(r.experiment_id)
    if not r.fits:
        raise ValueError("fit models first")
    t, y, valid, noise = _arrays(r, e)
    hold = valid & (t > r.split.train_end_t)
    amp = None
    if e.system_type == "pendulum":
        amp = {"deg": float(np.degrees(np.max(np.abs(y[valid]))))}
    c = compare_fits(r.fits, t[hold], noise, r.calibration, amp)
    r.comparison = c
    storage.save_run(r)
    return c


def discover(run_id: str) -> dict:
    from .discover import discover_pendulum

    r = storage.load_run(run_id)
    e = storage.load_experiment(r.experiment_id)
    if e.system_type != "pendulum":
        return {"skipped": "equation discovery is implemented for pendulums only"}
    t, y, valid, _ = _arrays(r, e)
    train = valid & (t <= r.split.train_end_t)
    res = discover_pendulum(t[train], y[train])
    r.discovery = res
    storage.save_run(r)
    return res


# ---------- design / commit / reveal ----------
def design(experiment_id: str, model_names: list[str] | None = None, knob: str | None = None) -> ExperimentProposal:
    e = storage.load_experiment(experiment_id)
    src = source_run(e)
    if src is None:
        raise ValueError("no analysed exploration run to design from")
    knob = knob or DEFAULT_KNOB[e.system_type]
    fits = [f for f in src.fits if model_names is None or f.model in model_names]
    noise = src.series[PRIMARY_SERIES[e.system_type]].noise_sigma
    ref_knob = float(src.knobs.get(knob, 1.0))
    p = design_experiment(fits, knob, ref_knob, noise, source_run_id=src.id, max_value=e.knob_max)
    e.proposal = p
    storage.save_experiment(e)
    return p


def commit(experiment_id: str, model_names: list[str] | None = None, knob_setting: dict[str, float] | None = None) -> Commitment:
    e = storage.load_experiment(experiment_id)
    src = source_run(e)
    if src is None:
        raise ValueError("no analysed exploration run to commit from")
    if knob_setting is None:
        if e.proposal is None:
            raise ValueError("design an experiment first (or pass knob_setting)")
        knob_setting = {e.proposal.knob: e.proposal.value}
    knob, value = next(iter(knob_setting.items()))
    fits = [f for f in src.fits if model_names is None or f.model in model_names]
    noise = src.series[PRIMARY_SERIES[e.system_type]].noise_sigma
    c = build_commitment(e.id, fits, knob, float(src.knobs.get(knob, 1.0)), float(value), storage.now_iso(),
                         get_settings().conditioning_window_s, commitment_reference(src, e.system_type), noise, src.id)
    storage.save_commitment(c)
    e.commit_ids.append(c.id)
    storage.save_experiment(e)
    return c


def evaluate(commitment_id: str, run_id: str) -> Verdict:
    c = storage.load_commitment(commitment_id)
    r = storage.load_run(run_id)
    e = storage.load_experiment(r.experiment_id)
    if r.validity is None:
        check_validity(run_id)
        r = storage.load_run(run_id)
    t, y, valid, noise = _arrays(r, e)
    v, curves = evaluate_commitment(c, r.id, t, y, valid, noise, r.validity)
    r.verdict = v
    storage.save_run(r)
    if curves:
        _save_curves(r, {k: np.asarray(val) for k, val in curves.items() if k != "t"})
    return v


def render(run_id: str, model_names: list[str] | None = None) -> str:
    from .render import render_run

    r = storage.load_run(run_id)
    out = render_run(r, model_names)
    r.overlay_path = storage.rel_path(out)
    storage.save_run(r)
    return r.overlay_path


def list_model_specs(system_type: str) -> list[dict]:
    return [m.spec().model_dump() for m in models_for(system_type)]
