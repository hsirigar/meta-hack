"""Render synthetic experiment videos with known physics.

  uv run --project backend python scripts/make_synthetic.py --all            # demo set -> data/raw/synthetic/
  uv run --project backend python scripts/make_synthetic.py pendulum --angle 70 --bump-at 5 -o out.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ghostline.synthetic import SynthConfig, make

ROOT = Path(__file__).resolve().parents[1]

DEMO_SET = {
    "pendulum_8": SynthConfig(system="pendulum", angle_deg=8, seed=11),
    "pendulum_70": SynthConfig(system="pendulum", angle_deg=70, seed=12),
    "pendulum_70_bumped": SynthConfig(system="pendulum", angle_deg=70, seed=13, bump_at=5.0),
    "pendulum_70_zoomed": SynthConfig(system="pendulum", angle_deg=70, seed=14, zoom_at=6.0),
    "filters_1": SynthConfig(system="falling_drag", n_filters=1, seed=21),
    "filters_4": SynthConfig(system="falling_drag", n_filters=4, seed=22),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", nargs="?", choices=["pendulum", "falling_drag"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--outdir", type=Path, default=ROOT / "data" / "raw" / "synthetic")
    ap.add_argument("--angle", type=float, default=8.0)
    ap.add_argument("--filters", type=int, default=1)
    ap.add_argument("--truth", default="", help="truth model name (default nonlinear / quadratic)")
    ap.add_argument("--noise-px", type=float, default=0.6)
    ap.add_argument("--bump-at", type=float)
    ap.add_argument("--zoom-at", type=float)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.all:
        for name, cfg in DEMO_SET.items():
            out = a.outdir / f"{name}.mp4"
            make(out, cfg)
            print("wrote", out)
        return
    if not a.system or not a.out:
        ap.error("give a system and -o, or --all")
    cfg = SynthConfig(system=a.system, angle_deg=a.angle, n_filters=a.filters, truth_model=a.truth,
                      track_noise_px=a.noise_px, bump_at=a.bump_at, zoom_at=a.zoom_at, seed=a.seed)
    make(a.out, cfg)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
