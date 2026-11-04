"""Build the three demo scenes into data/demo/ (served with DEMO_MODE=1, zero uploads needed).

Real footage is used when present in data/raw/real/ with these names, else synthetic clips:
  pendulum_8.mp4  pendulum_70.mp4  pendulum_70_bumped.mp4  filters_1.mp4  filters_4.mp4

  uv run --project backend python scripts/precompute_demo.py [--real-dir data/raw/real] [--keep-cache]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "demo")
    ap.add_argument("--real-dir", type=Path, default=ROOT / "data" / "raw" / "real")
    ap.add_argument("--synthetic-dir", type=Path, default=ROOT / "data" / "raw" / "synthetic")
    ap.add_argument("--keep-cache", action="store_true", help="keep data/demo/cache (API responses)")
    a = ap.parse_args()

    cache_backup = None
    if a.out.exists():
        if a.keep_cache and (a.out / "cache").exists():
            cache_backup = Path(str(a.out) + ".cache_bak")
            shutil.rmtree(cache_backup, ignore_errors=True)
            shutil.move(str(a.out / "cache"), cache_backup)
        for p in (a.out / "commits").glob("*.json") if (a.out / "commits").exists() else []:
            p.chmod(0o644)
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    if cache_backup:
        shutil.move(str(cache_backup), a.out / "cache")
    os.environ["DATA_DIR"] = str(a.out)

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_synthetic import DEMO_SET  # noqa: E402

    from ghostline.agent.loop import run_task  # noqa: E402
    from ghostline import pipeline as P  # noqa: E402
    from ghostline.synthetic import make  # noqa: E402

    def clip(name: str) -> Path:
        real = a.real_dir / f"{name}.mp4"
        if real.exists():
            print(f"  using REAL footage {real}")
            return real
        syn = a.synthetic_dir / f"{name}.mp4"
        if not syn.exists():
            make(syn, DEMO_SET[name])
        print(f"  using synthetic {syn}")
        return syn

    def agent(task, **ids):
        last = None
        for ev in run_task(task, **ids):
            if ev["type"] in ("message", "error"):
                print(f"    [{ev['type']}] {ev['text']}")
            if ev["type"] == "done":
                last = ev.get("result")
        return last or {}

    T = time.time()
    print("Scene A: pendulum")
    ea = P.create_experiment("Pendulum A", "pendulum", "Small-angle pendulum: linear or nonlinear? The green disk taped in the plane of motion is a US quarter (24.26 mm diameter).", exp_id="exp_pendulum_a")
    r8 = P.add_run(ea.id, clip("pendulum_8"), {"release_angle_deg": 8}, run_id="run_a_8deg")
    agent("analyze", run_id=r8.id)
    res = agent("next", experiment_id=ea.id)
    cid = res.get("commitment_id")
    r70 = P.add_run(ea.id, clip("pendulum_70"), {"release_angle_deg": 70}, role="reveal", commitment_id=cid, run_id="run_a_70deg")
    agent("score", run_id=r70.id, commitment_id=cid)
    print(f"  ({time.time() - T:.0f}s)")

    print("Scene C: bumped camera")
    rb = P.add_run(ea.id, clip("pendulum_70_bumped"), {"release_angle_deg": 70}, role="reveal", commitment_id=cid, run_id="run_a_bumped")
    agent("score", run_id=rb.id, commitment_id=cid)
    print(f"  ({time.time() - T:.0f}s)")

    print("Scene B: coffee filters")
    eb = P.create_experiment("Coffee filters", "falling_drag", "Falling coffee filters: linear or quadratic drag? The green rectangle taped to the wall is a credit card (85.6 mm long side).", exp_id="exp_filters_b")
    r1 = P.add_run(eb.id, clip("filters_1"), {"n_filters": 1}, run_id="run_b_1filter")
    agent("analyze", run_id=r1.id)
    res = agent("next", experiment_id=eb.id)
    cidb = res.get("commitment_id")
    r4 = P.add_run(eb.id, clip("filters_4"), {"n_filters": 4}, role="reveal", commitment_id=cidb, run_id="run_b_4filters")
    agent("score", run_id=r4.id, commitment_id=cidb)
    small, large = a.real_dir / "mouse_small.mp4", a.real_dir / "mouse_large.mp4"
    if small.exists() and large.exists():
        print("Scene D: real footage (a computer mouse on its cable)")
        ed = P.create_experiment(
            "Real footage: mouse on a cable", "pendulum",
            "Real phone footage: a computer mouse hanging on its cable in front of a mesh chair back. The reference is "
            "an ID-1 card (85.6 mm long side, same size as a credit card) on the ledge below; its face is blurred for privacy. "
            "This rig can only swing to about 20° before the mouse leaves the frame.",
            exp_id="exp_real_mouse", knob_max=20.0)
        rs = P.add_run(ed.id, small, run_id="run_d_small")
        agent("analyze", run_id=rs.id)
        res = agent("next", experiment_id=ed.id)
        cidd = res.get("commitment_id")
        rl = P.add_run(ed.id, large, role="reveal", commitment_id=cidd, run_id="run_d_large")
        agent("score", run_id=rl.id, commitment_id=cidd)
        print(f"  ({time.time() - T:.0f}s)")

    js, jl = a.real_dir / "jlab_small.mp4", a.real_dir / "jlab_large.mp4"
    if js.exists() and jl.exists():
        print("Scene E: real footage (Jefferson Lab, two release angles)")
        ee = P.create_experiment(
            "Real footage: lab pendulum (Jefferson Lab)", "pendulum",
            "Public footage from Jefferson Lab's Frostbite Theater video \"Swinging Pendulums (Different Angles)\" "
            "(YouTube), cropped, with the video's on-screen labels blacked out. "
            "A white washer on a thread hangs from the clamp of a lab stand. Nothing of known size is in view, so use "
            "the horizontal silver bar at the top of the stand as a fixed landmark (concept \"silver bar\", known_size_m 0): angles and periods only. This clip swings "
            "about 7 degrees; the largest swing available in the same footage is about 25 degrees.",
            exp_id="exp_real_jlab", knob_max=25.0)
        rs = P.add_run(ee.id, js, run_id="run_e_small")
        agent("analyze", run_id=rs.id)
        res = agent("next", experiment_id=ee.id)
        cide = res.get("commitment_id")
        rl = P.add_run(ee.id, jl, role="reveal", commitment_id=cide, run_id="run_e_large")
        agent("score", run_id=rl.id, commitment_id=cide)
        print(f"  ({time.time() - T:.0f}s)")

    from ghostline.config import get_settings
    from ghostline.scene import scene_backend_label
    s = get_settings()
    seg = s.segment_backend
    (a.out / "manifest.json").write_text(json.dumps({
        "scene_backend": scene_backend_label(), "segment_backend": seg, "llm_backend": s.llm_backend,
        "agent_model": s.agent_model if s.llm_backend == "meta" else "scripted mock",
        "muse_spark_open": s.muse_spark_open_active, "built_at": time.strftime("%Y-%m-%d %H:%M %Z")}, indent=1))
    print(f"done in {time.time() - T:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
