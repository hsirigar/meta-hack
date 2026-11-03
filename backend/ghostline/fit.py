"""Model fitting (SPEC §10). Leakage guard: these functions take plain arrays only.

`fit_models` must only ever be given training samples (t <= train_end_t); `pipeline.fit_run`
does the slicing. Holdout scoring is a separate function that receives frozen parameters.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares
from statsmodels.stats.diagnostic import acorr_ljungbox

from .models.base import PhysModel, get_model
from .schemas import FitResult


def _vec_bounds(model: PhysModel):
    lo = np.array([p.lower for p in model.all_params], float)
    hi = np.array([p.upper for p in model.all_params], float)
    return lo, hi


def _unpack(model: PhysModel, vec) -> tuple[dict, list[float]]:
    return model.split_params(vec)


def _jacobian(f, x, lo, hi):
    f0 = f(x)
    J = np.empty((len(f0), len(x)))
    for j in range(len(x)):
        h = 1e-6 * max(abs(x[j]), 1e-3)
        xp = x.copy()
        if xp[j] + h > hi[j]:
            h = -h
        xp[j] += h
        J[:, j] = (f(xp) - f0) / h
    return J


def _ljung_box_p(r: np.ndarray) -> float:
    n = len(r)
    if n < 12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = acorr_ljungbox(r, lags=[min(10, n // 5)], return_df=True)
    return float(lb["lb_pvalue"].iloc[0])


def _solve(model, t, y, t0, x_start, lo, hi, f_scale, max_nfev=300):
    def res(v):
        phys, x0 = _unpack(model, v)
        return model.simulate(t, phys, x0, t0) - y

    x_start = np.clip(x_start, lo + 1e-9, hi - 1e-9)
    return least_squares(res, x_start, bounds=(lo, hi), method="trf", loss="soft_l1",
                         f_scale=f_scale, x_scale="jac", max_nfev=max_nfev), res


def fit_model(model: PhysModel, t: np.ndarray, y: np.ndarray, noise_sigma: float, run_id: str = "",
              n_starts: int = 5) -> FitResult:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < len(model.all_params) + 3:
        raise ValueError(f"not enough training samples ({len(t)}) to fit {model.name}")
    t0 = float(t[0])
    lo, hi = _vec_bounds(model)
    f_scale = max(3 * float(noise_sigma), 1e-9)
    best, best_res = None, None
    for start in model.initial_guesses(t, y)[:n_starts]:
        sol, res = _solve(model, t, y, t0, np.asarray(start, float), lo, hi, f_scale)
        if best is None or sol.cost < best.cost:
            best, best_res = sol, res
    v = best.x
    r = best_res(v)
    n, k = len(y), len(v)
    rss = float(np.sum(r ** 2))
    s2 = rss / max(n - k, 1)
    J = _jacobian(best_res, v, lo, hi)
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        std = np.full(k, np.nan)
    names = [p.name for p in model.all_params]
    params = {nm: float(val) for nm, val in zip(names, v)}
    params["t_ref"] = t0
    pstd = {nm: float(sv) for nm, sv in zip(names, std)}
    pstd["t_ref"] = 0.0
    ll = n * np.log(max(rss / n, 1e-300))
    return FitResult(model=model.name, run_id=run_id, params=params, param_std=pstd,
                     train_rmse=float(np.sqrt(rss / n)), aic=float(ll + 2 * k), bic=float(ll + k * np.log(n)),
                     residual_whiteness_p=_ljung_box_p(r), n_train=n)


def fit_models(t_train: np.ndarray, y_train: np.ndarray, model_names: list[str], noise_sigma: float,
               run_id: str = "") -> list[FitResult]:
    """Fit candidate models on TRAINING arrays only."""
    return [fit_model(get_model(m), t_train, y_train, noise_sigma, run_id) for m in model_names]


# ---------- prediction with frozen parameters ----------
def physics_of(fit: FitResult) -> dict:
    model = get_model(fit.model)
    return {p.name: fit.params[p.name] for p in model.physics_params}


def state_of(fit: FitResult) -> list[float]:
    model = get_model(fit.model)
    return [fit.params[p.name] for p in model.state_params]


def predict(fit: FitResult, t: np.ndarray) -> np.ndarray:
    model = get_model(fit.model)
    return model.simulate(np.asarray(t, float), physics_of(fit), state_of(fit), fit.params["t_ref"])


def score_holdout(fit: FitResult, t_hold: np.ndarray, y_hold: np.ndarray) -> FitResult:
    """Holdout RMSE from frozen params (no refitting). Returns a new FitResult."""
    if len(t_hold) == 0:
        return fit.model_copy(update={"holdout_rmse": None, "n_holdout": 0})
    r = predict(fit, t_hold) - np.asarray(y_hold, float)
    return fit.model_copy(update={"holdout_rmse": float(np.sqrt(np.mean(r ** 2))), "n_holdout": int(len(t_hold))})


def bootstrap_band(fit: FitResult, t_train: np.ndarray, y_train: np.ndarray, t_grid: np.ndarray,
                   noise_sigma: float, n_boot: int = 40, seed: int = 0) -> dict[str, list[float]]:
    """Residual bootstrap prediction band (2.5-97.5%). Uses training data only."""
    model = get_model(fit.model)
    rng = np.random.default_rng(seed)
    t_train = np.asarray(t_train, float)
    yhat = predict(fit, t_train)
    resid = np.asarray(y_train, float) - yhat
    lo, hi = _vec_bounds(model)
    names = [p.name for p in model.all_params]
    v0 = np.array([fit.params[n] for n in names])
    preds = []
    for _ in range(n_boot):
        yb = yhat + rng.choice(resid, size=len(resid), replace=True)
        sol, _ = _solve(model, t_train, yb, fit.params["t_ref"], v0, lo, hi, max(3 * noise_sigma, 1e-9), max_nfev=60)
        phys, x0 = model.split_params(sol.x)
        preds.append(model.simulate(t_grid, phys, x0, fit.params["t_ref"]))
    P = np.array(preds)
    return {"t": np.asarray(t_grid).tolist(), "lo": np.percentile(P, 2.5, axis=0).tolist(),
            "hi": np.percentile(P, 97.5, axis=0).tolist()}
