import numpy as np
import pytest

from ghostline import storage
from ghostline.config import get_settings
from ghostline.models.base import get_model
from ghostline.schemas import Calibration, Experiment, Run, SceneDescription, ReferenceObject, Split, TimeSeries


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv("SEGMENT_BACKEND", "fallback")
    monkeypatch.setenv("SCENE_BACKEND", "mock")
    get_settings.cache_clear()
    yield tmp_path / "data"
    get_settings.cache_clear()


def synth_series(system="pendulum", knob=8.0, noise=6e-4, duration=None, fps=60.0, seed=0, truth=None):
    rng = np.random.default_rng(seed)
    if system == "pendulum":
        m = get_model(truth or "pendulum_nonlinear")
        t = np.arange(0, duration or 12.0, 1 / fps)
        y = m.simulate(t, {"omega0": 4.0435, "gamma": 0.03}, [np.radians(knob), 0.0], 0.0)
    else:
        m = get_model(truth or "drag_quadratic")
        vt = 1.1 * knob ** m.exponent
        t_all = np.arange(0, 6, 1 / fps)
        y_all = m.simulate(t_all, {"v_t": vt}, [-0.3, 0.0], 0.0)
        vis = (y_all >= 0) & (y_all <= 2.2)
        t, y = t_all[vis], y_all[vis]
    return t, y + rng.normal(0, noise, len(t))


def make_array_run(system="pendulum", knob=8.0, role="exploration", exp_id=None, run_id=None, noise=6e-4, seed=0,
                   commitment_id=None):
    """Experiment + run with a measured series but no video (for numerics/leakage tests)."""
    t, y = synth_series(system, knob, noise, seed=seed)
    exp_id = exp_id or storage.new_id("exp")
    try:
        e = storage.load_experiment(exp_id)
    except KeyError:
        e = Experiment(id=exp_id, name="test", system_type=system)
    name = "theta" if system == "pendulum" else "y"
    knob_name = "release_angle_deg" if system == "pendulum" else "n_filters"
    ts = TimeSeries(name=name, unit="rad" if system == "pendulum" else "m", t=t.tolist(), y=y.tolist(),
                    y_smooth=y.tolist(), noise_sigma=noise, valid_mask=[True] * len(t))
    cw = 0.5
    split = Split(train_end_t=float(t[0] + (cw if role == "reveal" else 0.6 * (t[-1] - t[0]))), conditioning_window_s=cw)
    cal = Calibration(meters_per_pixel=0.00125, meters_per_pixel_std=0.0, method="reference_object",
                      pivot_px=(640, 70), length_px=480, length_m=0.6)
    scene = SceneDescription(system_type=system, tracked_concepts=["orange bob"],
                             reference_object=ReferenceObject(concept="green disk", known_size_m=0.02426))
    r = Run(id=run_id or storage.new_id("run"), experiment_id=exp_id, video_path="none.mp4", video_sha256="0",
            fps_nominal=60, frame_times=t.tolist(), series={name: ts}, split=split, knobs={knob_name: knob},
            role=role, calibration=cal, scene=scene, commitment_id=commitment_id)
    storage.save_run(r)
    e.run_ids.append(r.id)
    storage.save_experiment(e)
    return e, r
