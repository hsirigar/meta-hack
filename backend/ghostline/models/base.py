"""Model base class + registry.

Each model integrates an ODE for its state and returns the observed variable.
Integration uses scipy's LSODA (`odeint`) at rtol=1e-8 for speed; closed forms are used where exact.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from scipy.integrate import odeint

from ..schemas import ModelSpec, ParamSpec

G = 9.81
RTOL = 1e-8
ATOL = 1e-10


class PhysModel(ABC):
    name: ClassVar[str]
    system: ClassVar[str]
    latex: ClassVar[str]
    state_vars: ClassVar[list[str]]
    physics_params: ClassVar[list[ParamSpec]]
    state_params: ClassVar[list[ParamSpec]]  # initial state, named like "theta0", "dtheta0"
    knob: ClassVar[str]
    knob_transfer: ClassVar[str]
    observable: ClassVar[str]

    # ---- dynamics ----
    @abstractmethod
    def rhs(self, x: np.ndarray, t: float, p: dict) -> list[float]: ...

    def simulate_state(self, t: np.ndarray, p: dict, x0, t0: float) -> np.ndarray:
        """Integrate from (t0, x0) to times t (t >= t0 or t < t0 both allowed). Returns (len(t), n_state)."""
        t = np.asarray(t, float)
        out = np.empty((len(t), len(self.state_vars)))
        fwd = t >= t0
        for mask, sign in ((fwd, 1.0), (~fwd, -1.0)):
            if not mask.any():
                continue
            tt = t[mask]
            order = np.argsort(sign * tt)
            grid = np.concatenate([[t0], tt[order]])
            sol = odeint(self.rhs, np.asarray(x0, float), grid, args=(p,), rtol=RTOL, atol=ATOL, mxstep=50000)
            res = np.empty((len(tt), sol.shape[1]))
            res[order] = sol[1:]
            out[mask] = res
        return out

    def simulate(self, t: np.ndarray, p: dict, x0, t0: float) -> np.ndarray:
        """Observed variable (first state var) at times t."""
        return self.simulate_state(t, p, x0, t0)[:, 0]

    # ---- fitting helpers ----
    @property
    def all_params(self) -> list[ParamSpec]:
        return self.physics_params + self.state_params

    def split_params(self, vec) -> tuple[dict, list[float]]:
        n = len(self.physics_params)
        phys = {ps.name: float(v) for ps, v in zip(self.physics_params, vec[:n])}
        return phys, [float(v) for v in vec[n:]]

    @abstractmethod
    def initial_guesses(self, t: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        """Multi-start initial parameter vectors (physics + state)."""

    @abstractmethod
    def transfer(self, phys: dict, from_knob: float, to_knob: float) -> dict:
        """Map physics params fitted at one knob setting to another (SPEC §9 knob transfer)."""

    @abstractmethod
    def default_initial_state(self, knob_value: float) -> list[float]:
        """Initial state for experiment design at a given knob setting."""

    @abstractmethod
    def design_horizon(self, phys: dict, knob_value: float) -> float: ...

    def observed_mask(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Which simulated samples the camera would actually see (experiment design)."""
        return np.ones(len(t), bool)

    @abstractmethod
    def predicted_observables(self, phys: dict, knob_value: float) -> dict[str, float]: ...

    def spec(self) -> ModelSpec:
        return ModelSpec(name=self.name, latex=self.latex, params=self.all_params,
                         state_vars=self.state_vars, knob_transfer=self.knob_transfer,
                         observable=self.observable)


_REGISTRY: dict[str, PhysModel] = {}


def register(cls):
    _REGISTRY[cls.name] = cls()
    return cls


def get_model(name: str) -> PhysModel:
    from . import drag, pendulum  # noqa: F401  (populate registry)
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def models_for(system_type: str, include_p1: bool = True) -> list[PhysModel]:
    from . import drag, pendulum  # noqa: F401
    out = [m for m in _REGISTRY.values() if m.system == system_type]
    if not include_p1:
        out = [m for m in out if not getattr(m, "extended", False)]
    return out


DEFAULT_CANDIDATES = {
    "pendulum": ["pendulum_linear", "pendulum_nonlinear"],
    "falling_drag": ["drag_linear", "drag_quadratic"],
}
