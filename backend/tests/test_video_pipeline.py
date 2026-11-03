"""End-to-end on short synthetic videos: tracking, calibration, circle fit, validity."""
import numpy as np
import pytest

from ghostline import pipeline as P
from ghostline import storage
from ghostline.measure import fit_circle
from ghostline.synthetic import SynthConfig, make


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("vids")
    specs = {"clean": SynthConfig(angle_deg=40, duration=5, seed=1),
             "bump": SynthConfig(angle_deg=40, duration=5, seed=2, bump_at=2.5),
             "zoom": SynthConfig(angle_deg=40, duration=5, seed=3, zoom_at=2.5),
             "drop": SynthConfig(system="falling_drag", n_filters=1, seed=4)}
    out = {}
    for k, cfg in specs.items():
        out[k] = (d / f"{k}.mp4", make(d / f"{k}.mp4", cfg))
    return out


def test_circle_fit_recovers_pivot():
    rng = np.random.default_rng(0)
    a = np.linspace(-0.7, 0.7, 300)
    x = 640 + 480 * np.sin(a) + rng.normal(0, 0.5, 300)
    y = 70 + 480 * np.cos(a) + rng.normal(0, 0.5, 300)
    cx, cy, R, rms = fit_circle(x, y)
    assert abs(cx - 640) < 1 and abs(cy - 70) < 2 and abs(R - 480) < 2 and rms < 1


def _process(video, system):
    e = P.create_experiment("t", system)
    r = P.add_run(e.id, video)
    P.describe_scene(r.id)
    P.track_objects(r.id)
    P.measure(r.id)
    return P.check_validity(r.id), storage.load_run(r.id)


def test_clean_pendulum_video(videos):
    path, truth = videos["clean"]
    rep, r = _process(path, "pendulum")
    assert rep.status == "valid", rep.reasons
    assert r.calibration.meters_per_pixel == pytest.approx(1 / truth["px_per_m"], rel=0.04)
    assert r.calibration.length_px == pytest.approx(truth["length_px"], rel=0.01)
    assert r.frame_times[1] == pytest.approx(1 / 60, abs=1e-4)
    fits = P.fit_run(r.id, bootstrap=0)
    nl = next(f for f in fits if f.model == "pendulum_nonlinear")
    assert nl.params["omega0"] == pytest.approx(truth["params"]["omega0"], rel=3e-3)


def test_bumped_camera_is_invalid(videos):
    rep, _ = _process(videos["bump"][0], "pendulum")
    assert rep.status == "invalid"
    assert any(c.name == "camera_jump" and c.status == "invalid" for c in rep.checks)


def test_zoom_is_invalid(videos):
    rep, _ = _process(videos["zoom"][0], "pendulum")
    assert rep.status == "invalid"
    assert any(c.name in ("scale_changed", "camera_drift") and c.status == "invalid" for c in rep.checks)


def test_drop_video_measures_terminal_velocity(videos):
    path, truth = videos["drop"]
    rep, r = _process(path, "falling_drag")
    assert rep.status in ("valid", "warning"), rep.reasons
    fits = P.fit_run(r.id, bootstrap=0)
    for f in fits:
        assert f.params["v_t"] == pytest.approx(truth["params"]["v_t"], rel=0.05)
