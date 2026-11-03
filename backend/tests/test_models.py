import numpy as np
import pytest

from ghostline.fit import fit_models, score_holdout
from ghostline.models.base import get_model
from ghostline.models.pendulum import PendulumLinear, measure_period

from conftest import synth_series


def test_linear_closed_form_matches_ode():
    m = get_model("pendulum_linear")
    p = {"omega0": 4.0, "gamma": 0.05}
    t = np.linspace(-1, 5, 400)
    closed = m.simulate(t, p, [0.2, 0.1], 0.5)
    ode = super(PendulumLinear, m).simulate_state(t, p, [0.2, 0.1], 0.5)[:, 0]
    assert np.max(np.abs(closed - ode)) < 1e-6


def test_period_ratio_at_70_degrees():
    m = get_model("pendulum_nonlinear")
    p = {"omega0": 4.0, "gamma": 0.0}
    t = np.linspace(0, 10, 20000)
    T70 = measure_period(t, m.simulate(t, p, [np.radians(70), 0], 0))
    T8 = measure_period(t, m.simulate(t, p, [np.radians(8), 0], 0))
    T0 = 2 * np.pi / 4.0
    from scipy.special import ellipk
    exact = 2 * ellipk(np.sin(np.radians(35)) ** 2) / np.pi  # exact large-amplitude period ratio
    assert T70 / T0 == pytest.approx(exact, abs=1e-3)
    assert T70 / T0 > 1.09  # ~10% longer than linear
    assert T8 / T0 == pytest.approx(1.0012, abs=0.0005)   # ~0.1%


@pytest.mark.parametrize("model", ["pendulum_linear", "pendulum_nonlinear"])
def test_pendulum_recovers_parameters(model):
    t, y = synth_series("pendulum", 40, noise=6e-4, truth=model, seed=3)
    tr = t <= 7.2
    f = fit_models(t[tr], y[tr], [model], 6e-4)[0]
    assert f.params["omega0"] == pytest.approx(4.0435, rel=2e-3)
    assert f.params["gamma"] == pytest.approx(0.03, abs=3e-3)
    f = score_holdout(f, t[~tr], y[~tr])
    assert f.holdout_rmse < 2 * 6e-4


@pytest.mark.parametrize("model,k", [("drag_quadratic", 4), ("drag_linear", 2)])
def test_drag_recovers_terminal_velocity(model, k):
    m = get_model(model)
    t, y = synth_series("falling_drag", k, noise=5e-4, truth=model, seed=4)
    f = fit_models(t, y, [model], 5e-4)[0]
    assert f.params["v_t"] == pytest.approx(1.1 * k ** m.exponent, rel=0.02)


def test_knob_transfer():
    assert get_model("drag_linear").transfer({"v_t": 1.0}, 1, 4)["v_t"] == pytest.approx(4.0)
    assert get_model("drag_quadratic").transfer({"v_t": 1.0}, 1, 4)["v_t"] == pytest.approx(2.0)
