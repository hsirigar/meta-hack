"""Falling body with drag. Observed variable: y (m), measured downward along gravity."""
from __future__ import annotations

import numpy as np

from ..schemas import ParamSpec
from .base import G, PhysModel, register

DROP_HEIGHT_M = 2.0  # visible fall distance
RELEASE_ABOVE_M = 0.3  # filters are released just above the top of the frame


class _DragBase(PhysModel):
    system = "falling_drag"
    state_vars = ["y", "v"]
    physics_params = [ParamSpec(name="v_t", unit="m/s", lower=0.05, upper=30.0)]
    state_params = [ParamSpec(name="y0", unit="m", lower=-20.0, upper=20.0),
                    ParamSpec(name="v0", unit="m/s", lower=-2.0, upper=30.0)]
    knob = "n_filters"
    observable = "terminal_velocity_mps"
    exponent = 1.0  # v_t(k) = k**exponent * v_t(1)

    def transfer(self, phys, from_knob, to_knob):
        return {"v_t": phys["v_t"] * (to_knob / from_knob) ** self.exponent}

    def initial_guesses(self, t, y):
        n = len(t)
        half = slice(n // 2, n)
        vt = float(np.polyfit(t[half], y[half], 1)[0]) if n >= 6 else 1.0
        vt = max(vt, 0.1)
        k = min(n, 7)
        v0 = float(np.polyfit(t[:k] - t[0], y[:k], 1)[0]) if k >= 3 else vt
        starts = []
        for fv, f0 in ((1.0, 1.0), (0.8, 1.0), (1.25, 1.0), (1.5, 0.7), (1.0, 0.9)):
            starts.append(np.array([vt * fv, float(y[0]), max(min(v0 * f0, 29.0), -1.9)]))
        return starts

    def default_initial_state(self, knob_value):
        return [-RELEASE_ABOVE_M, 0.0]

    def observed_mask(self, t, y):
        return (y >= 0) & (y <= DROP_HEIGHT_M)

    def design_horizon(self, phys, knob_value):
        t = np.linspace(0, 10, 2001)
        y = self.simulate(t, phys, self.default_initial_state(knob_value), 0.0)
        i = int(np.searchsorted(y, DROP_HEIGHT_M))
        return float(t[min(i, len(t) - 1)])

    def predicted_observables(self, phys, knob_value):
        return {"terminal_velocity_mps": float(phys["v_t"]),
                "fall_time_2m_s": self.design_horizon(phys, knob_value)}


@register
class DragLinear(_DragBase):
    name = "drag_linear"
    latex = r"\dot v = g\,(1 - v/v_t)"
    knob_transfer = "v_t(k filters) = k * v_t(1)"
    exponent = 1.0

    def rhs(self, x, t, p):
        return [x[1], G * (1 - x[1] / p["v_t"])]

    def simulate_state(self, t, p, x0, t0):
        vt = p["v_t"]
        dt = np.asarray(t, float) - t0
        tau = vt / G
        e = np.exp(-dt / tau)
        v = vt + (x0[1] - vt) * e
        y = x0[0] + vt * dt + (x0[1] - vt) * tau * (1 - e)
        return np.stack([y, v], axis=1)


@register
class DragQuadratic(_DragBase):
    name = "drag_quadratic"
    latex = r"\dot v = g\,(1 - (v/v_t)^2)"
    knob_transfer = "v_t(k filters) = sqrt(k) * v_t(1)"
    exponent = 0.5

    def rhs(self, x, t, p):
        v = x[1]
        return [v, G * (1 - v * abs(v) / p["v_t"] ** 2)]
