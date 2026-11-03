"""Evaluation harness (SPEC §17). Synthetic clips with known truth -> parameter recovery,
model-selection accuracy, ambiguity detection, validity detection. Real clips in data/raw/real/ are
reported too (selection + validity, no ground truth). Writes eval/results.{md,json}.

  uv run --project backend python scripts/eval.py [--n 50]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cases(n: int, rng: np.random.Generator):
    out = []
    per = max(n // 5, 1)
    for i in range(per):  # small-angle pendulum: should be ambiguous
        out.append(dict(kind="pendulum_small", system="pendulum", angle_deg=float(rng.uniform(5, 10)),
                        length_m=float(rng.uniform(0.45, 0.7)), truth_model="pendulum_nonlinear"))
    for i in range(per):  # large-angle: nonlinear should win
        out.append(dict(kind="pendulum_large", system="pendulum", angle_deg=float(rng.uniform(45, 75)),
                        length_m=float(rng.uniform(0.45, 0.7)), truth_model="pendulum_nonlinear"))
    for i in range(per // 2):  # linear truth at large angle: linear should win (control)
        out.append(dict(kind="pendulum_linear_truth", system="pendulum", angle_deg=float(rng.uniform(45, 70)),
                        length_m=float(rng.uniform(0.45, 0.7)), truth_model="pendulum_linear"))
    for i in range(per):  # drag
        k = int(rng.choice([1, 3, 4]))
        tm = str(rng.choice(["drag_quadratic", "drag_linear"])) if k > 1 else "drag_quadratic"
        out.append(dict(kind=f"drag_{'single' if k == 1 else 'stack'}", system="falling_drag", n_filters=k,
                        vt1=float(rng.uniform(0.9, 1.3)), truth_model=tm))
    m = n - len(out)
    for i in range(max(m, 0)):  # invalid recordings
        bump = i % 2 == 0
        out.append(dict(kind="invalid_bump" if bump else "invalid_zoom", system="pendulum",
                        angle_deg=float(rng.uniform(20, 60)), truth_model="pendulum_nonlinear",
                        bump_at=4.0 if bump else None, zoom_at=None if bump else 4.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--real-dir", type=Path, default=ROOT / "data" / "raw" / "real")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="ghostline_eval_"))
    os.environ["DATA_DIR"] = str(work / "data")
    os.environ.setdefault("LLM_BACKEND", "mock")

    from ghostline import pipeline as P
    from ghostline import storage
    from ghostline.models.base import DEFAULT_CANDIDATES, get_model
    from ghostline.synthetic import SynthConfig, make

    rng = np.random.default_rng(a.seed)
    rows = []
    T = time.time()
    for i, c in enumerate(cases(a.n, rng)):
        cfg = SynthConfig(system=c["system"], truth_model=c["truth_model"], seed=1000 + i,
                          angle_deg=c.get("angle_deg", 8), length_m=c.get("length_m", 0.6),
                          n_filters=c.get("n_filters", 1), vt1=c.get("vt1", 1.1),
                          bump_at=c.get("bump_at"), zoom_at=c.get("zoom_at"),
                          duration=8.0 if c["system"] == "pendulum" else None)
        vid = work / f"clip_{i}.mp4"
        truth = make(vid, cfg)
        e = P.create_experiment(c["kind"], c["system"])
        r = P.add_run(e.id, vid, truth["knobs"])
        row = {**c, "i": i}
        try:
            P.describe_scene(r.id); P.track_objects(r.id); P.measure(r.id)
            v = P.check_validity(r.id)
            row["validity"] = v.status
            if v.status != "invalid":
                fits = P.fit_run(r.id, DEFAULT_CANDIDATES[c["system"]], bootstrap=0)
                cmp_ = P.compare_run(r.id)
                row["status"], row["best"] = cmp_.status, cmp_.best
                row["max_sep_sigma"] = cmp_.max_separation_sigma
                tf = next(f for f in fits if f.model == c["truth_model"])
                m = get_model(c["truth_model"])
                for p in m.physics_params:
                    est, true = tf.params[p.name], truth["params"][p.name]
                    row[f"err_{p.name}_pct"] = 100 * (est - true) / true if true else None
        except Exception as ex:  # record failures honestly
            row["error"] = f"{type(ex).__name__}: {ex}"
        rows.append(row)
        print(f"[{time.time() - T:5.0f}s] {i:2d} {c['kind']:22s} validity={row.get('validity')} "
              f"status={row.get('status')} best={row.get('best')} {row.get('error', '')}")

    def frac(sel, cond):
        s = [r for r in rows if sel(r)]
        return (sum(1 for r in s if cond(r)), len(s))

    def pct(t):
        return f"{t[0]}/{t[1]} ({100 * t[0] / t[1]:.0f}%)" if t[1] else "n/a"

    summary = {
        "ambiguity_detected_small_angle": frac(lambda r: r["kind"] == "pendulum_small", lambda r: r.get("status") == "ambiguous"),
        "nonlinear_selected_large_angle": frac(lambda r: r["kind"] == "pendulum_large", lambda r: r.get("best") == "pendulum_nonlinear"),
        "linear_selected_when_true": frac(lambda r: r["kind"] == "pendulum_linear_truth", lambda r: r.get("best") == "pendulum_linear"),
        "drag_single_ambiguous": frac(lambda r: r["kind"] == "drag_single", lambda r: r.get("status") == "ambiguous"),
        "drag_stack_correct_model": frac(lambda r: r["kind"] == "drag_stack", lambda r: r.get("best") == r["truth_model"]),
        "invalid_detected": frac(lambda r: r["kind"].startswith("invalid"), lambda r: r.get("validity") == "invalid"),
        "false_invalid_on_clean": frac(lambda r: not r["kind"].startswith("invalid"), lambda r: r.get("validity") == "invalid"),
        "errors": sum(1 for r in rows if "error" in r),
    }

    def med_abs(key, kinds):
        v = [abs(r[key]) for r in rows if r["kind"] in kinds and r.get(key) is not None]
        return (float(np.median(v)), float(np.percentile(v, 90)), len(v)) if v else None

    recovery = {
        "pendulum omega0": med_abs("err_omega0_pct", {"pendulum_small", "pendulum_large", "pendulum_linear_truth"}),
        "pendulum gamma": med_abs("err_gamma_pct", {"pendulum_small", "pendulum_large", "pendulum_linear_truth"}),
        "drag v_t": med_abs("err_v_t_pct", {"drag_single", "drag_stack"}),
    }

    real = []
    if a.real_dir.exists():
        for vid in sorted(a.real_dir.glob("*.mp4")) + sorted(a.real_dir.glob("*.mov")):
            system = "pendulum" if "pend" in vid.stem else "falling_drag"
            e = P.create_experiment(vid.stem, system)
            r = P.add_run(e.id, vid)
            row = {"clip": vid.name}
            try:
                P.describe_scene(r.id); P.track_objects(r.id); P.measure(r.id)
                v = P.check_validity(r.id)
                row["validity"] = v.status
                row["reasons"] = v.reasons
                if v.status != "invalid":
                    fits = P.fit_run(r.id, bootstrap=0)
                    cmp_ = P.compare_run(r.id)
                    noise = storage.load_run(r.id).series[next(iter(storage.load_run(r.id).series))].noise_sigma
                    row.update(status=cmp_.status, best=cmp_.best,
                               holdout_rmse_sigma={f.model: f.holdout_rmse / noise for f in fits})
            except Exception as ex:
                row["error"] = str(ex)
            real.append(row)

    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": summary, "recovery_abs_pct_median_p90_n": recovery,
                                                  "rows": rows, "real": real}, indent=1, default=str))
    lines = ["# Ghostline evaluation", "",
             f"Synthetic clips: {len(rows)} (rendered videos → fallback tracker → full pipeline; seed {a.seed}). "
             "Generated by `scripts/eval.py`.", "",
             "| Check | Result |", "|---|---|"]
    names = {"ambiguity_detected_small_angle": "Says *inconclusive* for small-angle pendulums (5–10°)",
             "nonlinear_selected_large_angle": "Picks nonlinear at 45–75° (nonlinear truth)",
             "linear_selected_when_true": "Picks linear when linear is the truth (control)",
             "drag_single_ambiguous": "Says *inconclusive* for a single coffee filter",
             "drag_stack_correct_model": "Picks the true drag law for 3–4 stacked filters",
             "invalid_detected": "Rejects bumped / zoomed recordings as INVALID",
             "false_invalid_on_clean": "False INVALID on clean recordings (lower is better)"}
    for k, label in names.items():
        lines.append(f"| {label} | {pct(summary[k])} |")
    lines += ["", "| Parameter recovery (true model) | median abs error | p90 | n |", "|---|---|---|---|"]
    for k, v in recovery.items():
        if v:
            lines.append(f"| {k} | {v[0]:.2f}% | {v[1]:.2f}% | {v[2]} |")
    lines += ["", f"Pipeline errors: {summary['errors']}.", ""]
    if real:
        lines += ["## Real footage", "", "| Clip | Validity | Status | Best |", "|---|---|---|---|"]
        for r in real:
            lines.append(f"| {r['clip']} | {r.get('validity')} | {r.get('status', '—')} | {r.get('best', '—')} |")
    else:
        lines += ["_No real clips yet (place them in `data/raw/real/`)._"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
