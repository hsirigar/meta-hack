"""Model comparison + ambiguity rule (SPEC §10)."""
from __future__ import annotations

import numpy as np

from .config import get_settings
from .fit import physics_of, predict
from .models.base import G
from .schemas import Calibration, Comparison, FitResult


def compare_fits(fits: list[FitResult], t_hold: np.ndarray, noise_sigma: float,
                 calibration: Calibration | None = None, amplitude: float | None = None) -> Comparison:
    s = get_settings()
    if not fits:
        raise ValueError("no fits to compare")
    ranked = sorted(fits, key=lambda f: f.bic)
    facts: dict = {"noise_sigma": noise_sigma, "models": {}}
    for f in fits:
        m = {"params": {k: round(v, 6) for k, v in f.params.items() if k != "t_ref"},
             "param_std": {k: round(v, 6) for k, v in f.param_std.items() if k != "t_ref"},
             "train_rmse": f.train_rmse, "holdout_rmse": f.holdout_rmse,
             "holdout_rmse_sigma": (f.holdout_rmse / noise_sigma) if f.holdout_rmse is not None else None,
             "bic": f.bic, "aic": f.aic, "ljung_box_p": f.residual_whiteness_p}
        if "omega0" in f.params:
            m["implied_length_m"] = G / f.params["omega0"] ** 2
        facts["models"][f.model] = m
    if calibration is not None and calibration.length_m:
        facts["measured_length_m"] = calibration.length_m
    if amplitude is not None:
        facts["amplitude"] = amplitude

    if len(ranked) == 1:
        return Comparison(fits=fits, best=ranked[0].model, status="decisive", max_separation_sigma=0.0,
                          explanation_facts=facts)

    a, b = ranked[0], ranked[1]
    dbic = b.bic - a.bic
    ha, hb = a.holdout_rmse or np.nan, b.holdout_rmse or np.nan
    ratio = float(max(ha, hb) / max(min(ha, hb), 1e-300)) if np.isfinite(ha) and np.isfinite(hb) else float("nan")
    t_hold = np.asarray(t_hold, float)
    if len(t_hold):
        sep = float(np.max(np.abs(predict(a, t_hold) - predict(b, t_hold))) / max(noise_sigma, 1e-12))
    else:
        sep = 0.0
    # BIC assumes independent residuals. On real footage residuals are often autocorrelated (Ljung-Box p < 0.01),
    # which inflates |dBIC|; then only the predictive criteria (holdout ratio, separation) can call it decisive.
    pa, pb = a.residual_whiteness_p, b.residual_whiteness_p
    bic_unreliable = all(p is not None and np.isfinite(p) and p < 0.01 for p in (pa, pb))
    bic_ok = abs(dbic) < s.ambig_dbic or bic_unreliable
    ambiguous = bic_ok and (ratio < s.ambig_rmse_ratio or not np.isfinite(ratio)) and sep < s.ambig_sep_sigma
    best = None
    if not ambiguous:
        with_h = [f for f in fits if f.holdout_rmse is not None]
        best = min(with_h, key=lambda f: f.holdout_rmse).model if with_h else a.model
    facts.update({"top_models": [a.model, b.model], "delta_bic": dbic, "holdout_rmse_ratio": ratio,
                  "max_separation_sigma": sep, "bic_ignored_autocorrelated_residuals": bic_unreliable,
                  "thresholds": {"delta_bic": s.ambig_dbic, "rmse_ratio": s.ambig_rmse_ratio, "sep_sigma": s.ambig_sep_sigma},
                  "rule": "ambiguous if |dBIC| < 6 and holdout RMSE ratio < 1.2 and max separation < 3 sigma "
                          "(the dBIC test is skipped when both models' residuals are autocorrelated, Ljung-Box p < 0.01)"})
    return Comparison(fits=fits, best=best, status="ambiguous" if ambiguous else "decisive",
                      max_separation_sigma=sep, explanation_facts=facts)
