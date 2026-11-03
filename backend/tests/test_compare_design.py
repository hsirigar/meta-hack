import numpy as np

from ghostline.compare import compare_fits
from ghostline.design import design_experiment
from ghostline.fit import fit_models, score_holdout

from conftest import synth_series


def _fit(system, knob, names, noise):
    t, y = synth_series(system, knob, noise=noise, seed=5)
    te = t[0] + 0.6 * (t[-1] - t[0])
    tr = t <= te
    fits = [score_holdout(f, t[~tr], y[~tr]) for f in fit_models(t[tr], y[tr], names, noise)]
    return fits, t[~tr]


def test_small_angle_is_ambiguous_large_angle_decisive():
    names = ["pendulum_linear", "pendulum_nonlinear"]
    fits, th = _fit("pendulum", 8, names, 1.2e-3)
    assert compare_fits(fits, th, 1.2e-3).status == "ambiguous"
    fits, th = _fit("pendulum", 70, names, 1.2e-3)
    c = compare_fits(fits, th, 1.2e-3)
    assert c.status == "decisive" and c.best == "pendulum_nonlinear"


def test_design_recommends_large_angle_and_four_filters():
    fits, _ = _fit("pendulum", 8, ["pendulum_linear", "pendulum_nonlinear"], 1.2e-3)
    p = design_experiment(fits, "release_angle_deg", 8, 1.2e-3)
    curve = dict(p.discrimination_curve)
    assert 65 <= p.value <= 75
    assert curve[10.0] < 3 < curve[p.value]
    fits, _ = _fit("falling_drag", 1, ["drag_linear", "drag_quadratic"], 1e-3)
    p = design_experiment(fits, "n_filters", 1, 1e-3)
    assert p.value == 4
    assert dict(p.discrimination_curve)[1.0] < 3
