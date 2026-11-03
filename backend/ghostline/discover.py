"""Equation discovery (P1, supporting evidence only).

PySINDy smoothed finite differences for derivatives, then sparse regression on the library
{theta, theta^3, sin theta, dtheta, dtheta|dtheta|} for theta'': best subset (<=3 terms) by BIC,
with STLSQ kept as a reference. Best-subset avoids huge cancelling coefficients on collinear terms.
At small amplitude theta and sin(theta) are nearly collinear, so discovery is (honestly) ambiguous there.
"""
from __future__ import annotations

import numpy as np

NAMES = ["theta", "theta^3", "sin(theta)", "dtheta", "dtheta|dtheta|"]


def _library(th, dth):
    return np.column_stack([th, th ** 3, np.sin(th), dth, dth * np.abs(dth)])


def stlsq(A: np.ndarray, b: np.ndarray, threshold: float, iters: int = 10) -> np.ndarray:
    scale = np.linalg.norm(A, axis=0) + 1e-12
    An = A / scale
    xi = np.linalg.lstsq(An, b, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(xi) < threshold * np.max(np.abs(xi))
        xi[small] = 0.0
        big = ~small
        if not big.any():
            break
        xi[big] = np.linalg.lstsq(An[:, big], b, rcond=None)[0]
    return xi / scale


def best_subset(A: np.ndarray, b: np.ndarray, max_terms: int = 3) -> np.ndarray:
    from itertools import combinations

    n, p = A.shape
    best, best_score = np.zeros(p), np.inf
    for k in range(1, max_terms + 1):
        for cols in combinations(range(p), k):
            cols = list(cols)
            x, *_ = np.linalg.lstsq(A[:, cols], b, rcond=None)
            rss = float(np.sum((A[:, cols] @ x - b) ** 2))
            score = n * np.log(max(rss / n, 1e-300)) + k * np.log(n) * 4  # strong sparsity prior
            if score < best_score:
                best_score, best = score, np.zeros(p)
                best[cols] = x
    return best


def discover_pendulum(t: np.ndarray, theta: np.ndarray, threshold: float = 0.05) -> dict:
    with np.errstate(all="ignore"):  # numpy+Accelerate emits spurious matmul FP warnings on macOS
        return _discover(t, theta, threshold)


def _discover(t, theta, threshold):
    import pysindy as ps

    t = np.asarray(t, float)
    tu = np.linspace(t[0], t[-1], len(t))
    thu = np.interp(tu, t, np.asarray(theta, float))
    dt = float(tu[1] - tu[0])
    sfd = ps.SmoothedFiniteDifference(smoother_kws={"window_length": 11, "polyorder": 3})
    dth = np.asarray(sfd(thu, t=dt)).ravel()
    ddth = np.asarray(sfd(dth, t=dt)).ravel()
    trim = slice(10, len(tu) - 10)  # derivative edge effects
    Theta = _library(thu, dth)[trim]
    target = ddth[trim]
    coef = best_subset(Theta, target)
    pred = Theta @ coef
    r2 = 1 - float(np.sum((target - pred) ** 2)) / max(float(np.sum((target - target.mean()) ** 2)), 1e-300)
    terms = {n: float(c) for n, c in zip(NAMES, coef) if c != 0}
    eq = "θ'' = " + " ".join(f"{'+' if c >= 0 else '−'} {abs(c):.4g}·{n}" for n, c in terms.items())
    amp = float(np.max(np.abs(thu)))
    return {"equation": eq, "terms": terms, "r2": r2, "library": NAMES, "threshold": threshold,
            "amplitude_deg": float(np.degrees(amp)),
            "method": "PySINDy smoothed finite differences + best-subset sparse regression (BIC)",
            "caveat": "theta and sin(theta) are nearly collinear below ~20 degrees" if amp < np.radians(20) else ""}
