"""Pendulum models. Observed variable: theta (rad), 0 = hanging straight down, + = to image right."""
from __future__ import annotations

import numpy as np

from ..schemas import ParamSpec
from .base import G, PhysModel, register

_PHYS = [ParamSpec(name="omega0", unit="rad/s", lower=0.5, upper=30.0),
         ParamSpec(name="gamma", unit="1/s", lower=0.0, upper=5.0)]
_STATE = [ParamSpec(name="theta0", unit="rad", lower=-2.2, upper=2.2),
          ParamSpec(name="dtheta0", unit="rad/s", lower=-30.0, upper=30.0)]


def zero_crossings(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolated times of upward zero crossings (after removing the mean offset)."""
    y = np.asarray(y, float) - np.median(y)
    s = np.sign(y)
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    out = []
    for i in idx:
        y0, y1 = y[i], y[i + 1]
        if y1 == y0:
            continue
        out.append(t[i] + (t[i + 1] - t[i]) * (-y0) / (y1 - y0))
    return np.array(out)


CONDITIONING_S = 0.5    # matches settings.conditioning_window_s (the reveal protocol)
PERIOD_WINDOW_S = 10.0  # period observable = mean period over this long after the conditioning window


def measure_period(t: np.ndarray, y: np.ndarray, max_periods: int | None = None) -> float | None:
    zc = zero_crossings(np.asarray(t), np.asarray(y))
    if max_periods:
        zc = zc[: max_periods + 1]
    if len(zc) < 2:
        return None
    return float(np.mean(np.diff(zc)))


def _guess_omega_gamma(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(observed angular frequency, damping rate, amplitude)."""
    period = measure_period(t, y)
    if period is None or period <= 0:
        # fall back to FFT on a uniform grid
        tu = np.linspace(t[0], t[-1], len(t))
        yu = np.interp(tu, t, y - np.mean(y))
        f = np.fft.rfftfreq(len(tu), tu[1] - tu[0])
        k = np.argmax(np.abs(np.fft.rfft(yu))[1:]) + 1
        period = 1.0 / max(f[k], 1e-3)
    w = 2 * np.pi / period
    # damping from the decay of |y| peaks per half period
    yc = y - np.median(y)
    n_per = max(int(round(period / max(np.median(np.diff(t)), 1e-4))), 4)
    peaks_t, peaks_a = [], []
    for s in range(0, len(yc) - n_per // 2, max(n_per // 2, 1)):
        seg = np.abs(yc[s: s + n_per // 2 + 1])
        j = int(np.argmax(seg))
        peaks_t.append(t[s + j]); peaks_a.append(seg[j])
    peaks_t, peaks_a = np.array(peaks_t), np.array(peaks_a)
    gamma = 0.02
    if len(peaks_a) >= 3 and np.all(peaks_a > 0):
        slope = np.polyfit(peaks_t, np.log(peaks_a), 1)[0]
        gamma = float(np.clip(-slope, 0.0, 2.0))
    amp = float(np.max(np.abs(yc[: max(n_per, 2)])))
    return w, gamma, amp


def _initial_rate(t: np.ndarray, y: np.ndarray) -> float:
    n = min(len(t), 7)
    if n < 3:
        return 0.0
    return float(np.polyfit(t[:n] - t[0], y[:n], 2)[1])


class _PendulumBase(PhysModel):
    system = "pendulum"
    state_vars = ["theta", "dtheta"]
    physics_params = _PHYS
    state_params = _STATE
    knob = "release_angle_deg"
    knob_transfer = "omega0 and gamma are properties of the pendulum and stay fixed; the new initial state comes from the conditioning window"
    observable = "period_s"
    amplitude_correction = False  # whether omega0 guess should undo the amplitude-period effect

    def initial_guesses(self, t, y):
        w_obs, gamma, amp = _guess_omega_gamma(t, y)
        w0 = w_obs
        if self.amplitude_correction:
            w0 = w_obs / max(1 - amp ** 2 / 16, 0.5)
        th0, dth0 = float(y[0]), _initial_rate(t, y)
        starts = []
        for fw, fg in ((1.0, 1.0), (0.98, 1.0), (1.02, 1.0), (1.0, 0.3), (1.0, 3.0)):
            starts.append(np.array([w0 * fw, min(max(gamma * fg, 1e-4), 4.9), th0, dth0]))
        return starts

    def transfer(self, phys, from_knob, to_knob):
        return dict(phys)

    def default_initial_state(self, knob_value):
        return [np.deg2rad(knob_value), 0.0]

    def design_horizon(self, phys, knob_value):
        return 6 * 2 * np.pi / phys["omega0"]

    def predicted_observables(self, phys, knob_value):
        """Mean period over the same window the reveal measures: PERIOD_WINDOW_S after the conditioning window
        (the swing decays, so this is shorter than the first-swing period for nonlinear models)."""
        x0 = self.default_initial_state(knob_value)
        T0 = 2 * np.pi / phys["omega0"]
        t = np.linspace(0, CONDITIONING_S + PERIOD_WINDOW_S, 8000)
        y = self.simulate(t, phys, x0, 0.0)
        w = t >= CONDITIONING_S
        per = measure_period(t[w], y[w])
        return {"period_s": float(per) if per else float(T0),
                "implied_length_m": float(G / phys["omega0"] ** 2)}


@register
class PendulumLinear(_PendulumBase):
    name = "pendulum_linear"
    latex = r"\ddot\theta + 2\gamma\dot\theta + \omega_0^2\,\theta = 0"

    def rhs(self, x, t, p):
        return [x[1], -2 * p["gamma"] * x[1] - p["omega0"] ** 2 * x[0]]

    def simulate_state(self, t, p, x0, t0):
        w0, g = p["omega0"], p["gamma"]
        if w0 <= g:
            return super().simulate_state(t, p, x0, t0)
        wd = np.sqrt(w0 ** 2 - g ** 2)
        a, b = x0[0], (x0[1] + g * x0[0]) / wd
        dt = np.asarray(t, float) - t0
        e, c, s = np.exp(-g * dt), np.cos(wd * dt), np.sin(wd * dt)
        th = e * (a * c + b * s)
        dth = e * ((-g * a + wd * b) * c + (-g * b - wd * a) * s)
        return np.stack([th, dth], axis=1)


@register
class PendulumNonlinear(_PendulumBase):
    name = "pendulum_nonlinear"
    latex = r"\ddot\theta + 2\gamma\dot\theta + \omega_0^2\,\sin\theta = 0"
    amplitude_correction = True

    def rhs(self, x, t, p):
        return [x[1], -2 * p["gamma"] * x[1] - p["omega0"] ** 2 * np.sin(x[0])]


@register
class PendulumNonlinearQuadDamp(_PendulumBase):
    name = "pendulum_nonlinear_quaddamp"
    latex = r"\ddot\theta + c\,\dot\theta|\dot\theta| + \omega_0^2\,\sin\theta = 0"
    amplitude_correction = True
    extended = True
    physics_params = [_PHYS[0], ParamSpec(name="c", unit="1/rad", lower=0.0, upper=5.0)]

    def rhs(self, x, t, p):
        return [x[1], -p["c"] * x[1] * abs(x[1]) - p["omega0"] ** 2 * np.sin(x[0])]

    def initial_guesses(self, t, y):
        starts = super().initial_guesses(t, y)
        w_obs, gamma, amp = _guess_omega_gamma(t, y)
        # quadratic damping ~ linear-equivalent: 2*gamma ≈ (8/3pi) c * amp * w
        c = 2 * gamma / max((8 / (3 * np.pi)) * max(amp, 1e-3) * w_obs, 1e-3)
        for s in starts:
            s[1] = min(max(c, 1e-4), 4.9)
        return starts
