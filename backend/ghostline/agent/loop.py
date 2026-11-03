"""Agent loop. Muse Wonder (Responses API, function tools) or a scripted mock policy.

Both emit the same event stream (the lab notebook):
  {"type": "status"|"thought"|"tool_call"|"tool_result"|"message"|"error"|"done", ...}
Every step is persisted as a trace so demos replay offline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from .. import storage
from ..clients.cache import cached_call
from ..config import get_settings
from . import tools

PROMPT = (Path(__file__).parent / "system_prompt.md").read_text()

TASKS = {
    "analyze": "Analyze run {run_id} of experiment {experiment_id} (a {system_type} experiment). Measure it, check validity, "
               "fit the candidate models, compare them on held-out data, and render the ghosts. Then tell me whether the "
               "evidence is decisive, in plain language.",
    "next": "Experiment {experiment_id} ({system_type}): what should I film next? Design the experiment that best separates "
            "the candidate models, commit the predictions before I film, and give me exact filming instructions.",
    "score": "I filmed the experiment you asked for. It is run {run_id}; score it against commitment {commitment_id}. "
             "Process the video, check validity, evaluate the commitment, render the ghosts and give me the verdict.",
}


def _ev(type_: str, **kw) -> dict:
    return {"type": type_, "ts": time.time(), **kw}


def trace_key(task: str, ids: dict) -> str:
    target = ids.get("run_id") or ids.get("experiment_id")
    return f"{task}_{target}"


def trace_path(key: str) -> Path:
    return storage.data_dir() / "traces" / f"{key}.json"


# ---------------- mock (scripted) policy ----------------
class _Mock:
    """Deterministic policy: the tool sequence a careful scientist would take; prose built from tool results."""

    label = "scripted policy (mock; set LLM_BACKEND=meta for Muse Wonder)"

    def __init__(self):
        self.results: dict[str, dict] = {}

    def call(self, name: str, args: dict, thought: str) -> Iterator[dict]:
        yield _ev("thought", text=thought)
        cid = f"call_{len(self.results)}_{name}"
        yield _ev("tool_call", id=cid, name=name, args=args)
        res = tools.execute(name, args)
        self.results[name] = res
        yield _ev("tool_result", id=cid, name=name, ok="error" not in res, result=res)

    # ---- tasks ----
    def analyze(self, run_id: str, experiment_id: str, system_type: str) -> Iterator[dict]:
        yield from self.call("describe_scene", {"run_id": run_id}, "First I need to know what's in the frame and what I can use as a ruler.")
        sc = self.results["describe_scene"]
        if "error" in sc:
            yield _ev("message", text=f"I couldn't understand the scene: {sc['error']}"); return
        yield from self.call("track_objects", {"run_id": run_id},
                             f"Tracking the {sc['tracked_concepts'][0]} and the {sc['reference_object']['concept']} ({sc['reference_object']['known_size_m'] * 1000:.2f} mm reference).")
        yield from self.call("measure", {"run_id": run_id}, "Converting pixels to physical units and fixing the train/holdout split before any fitting.")
        if "error" in self.results["measure"]:
            yield _ev("message", text=f"Measurement failed: {self.results['measure']['error']}"); return
        yield from self.call("check_validity", {"run_id": run_id}, "Before trusting these numbers, checking that the camera, scale and tracking held steady.")
        val = self.results["check_validity"]
        if val.get("status") == "invalid":
            reasons = [c["reason"] for c in val["checks"] if c["status"] == "invalid"]
            yield _ev("message", text="INVALID recording, so I won't fit anything. " + " ".join(reasons) +
                      " Please re-film with the phone fixed in place and the reference object visible the whole time.")
            return
        yield from self.call("list_models", {"system_type": system_type}, "Listing the candidate models for this system.")
        names = [m["name"] for m in self.results["list_models"]["models"] if not m["name"].endswith("quaddamp")]
        yield from self.call("fit_models", {"run_id": run_id, "model_names": names},
                             f"Fitting {' and '.join(names)} on the first part of the clip only. The rest is held out.")
        yield from self.call("compare_models", {"run_id": run_id}, "Comparing the models only on the data they never saw.")
        cmp_ = self.results["compare_models"]
        if system_type == "pendulum":
            yield from self.call("discover_equation", {"run_id": run_id}, "As a cross-check, letting sparse regression propose an equation from the data alone.")
        yield from self.call("render_ghosts", {"run_id": run_id, "model_names": names}, "Drawing each model's prediction over the video as a ghost.")
        meas = self.results["measure"]
        hs = cmp_.get("holdout_rmse_in_noise_sigma", {})
        if cmp_.get("status") == "ambiguous":
            amp = f" at {meas.get('amplitude_deg', 0):.1f}°" if "amplitude_deg" in meas else ""
            txt = (f"Inconclusive{amp}. Both models predict the held-out data equally well "
                   f"({', '.join(f'{k}: {v:.2f}σ' for k, v in hs.items())} RMSE), and their predictions never differ by more than "
                   f"{cmp_['max_separation_sigma']:.2f}σ of measurement noise (ΔBIC {cmp_['delta_bic']:.1f}). "
                   "This footage can't tell them apart. Ask me what to film next.")
        else:
            txt = (f"Decisive: {cmp_['best']} wins on held-out data "
                   f"({', '.join(f'{k}: {v:.2f}σ' for k, v in hs.items())} RMSE; models separate by up to {cmp_['max_separation_sigma']:.1f}σ).")
        if system_type == "pendulum" and cmp_.get("measured_length_m"):
            il = cmp_.get("implied_length_m", {})
            first = next(iter(il.values()), None)
            if first and 2 * meas.get("amplitude_deg", 0) >= 20:
                txt += f" Sanity check: implied length g/ω₀² = {first:.3f} m vs {cmp_['measured_length_m']:.3f} m measured from the video."
            elif first:
                txt += f" Implied length g/ω₀² = {first:.3f} m (the arc is too short to measure the string length precisely from the video)."
        yield _ev("message", text=txt)

    def next(self, experiment_id: str, system_type: str) -> Iterator[dict]:
        knob = "release_angle_deg" if system_type == "pendulum" else "n_filters"
        yield from self.call("design_experiment", {"experiment_id": experiment_id, "knob": knob},
                             "Searching over the setting I control for the experiment where the models disagree the most, relative to measurement noise.")
        d = self.results["design_experiment"]
        if "error" in d:
            yield _ev("message", text=f"I couldn't design an experiment: {d['error']}"); return
        yield from self.call("commit_predictions", {"experiment_id": experiment_id, "knob_setting": {knob: d["recommended_value"]}},
                             "Locking in every model's prediction now, before you film, so the result can't be fitted after the fact.")
        c = self.results["commit_predictions"]
        unit = "°" if knob == "release_angle_deg" else " filters"
        preds = c.get("predictions", {})
        obs = "; ".join(f"{k}: " + ", ".join(f"{kk} {vv:.3g}" for kk, vv in v.items() if not kk.endswith("_at_target"))
                        for k, v in preds.items())
        curve = {round(p["value"], 1): p["separation_sigma"] for p in d["discrimination_curve"]}
        txt = (f"Film this next: {d['recommended_value']:g}{unit}. There the models separate by {d['expected_separation_sigma']:.1f}σ "
               f"(max {d['expected_max_separation_sigma']:.0f}σ), versus {min(curve.values()):.1f}σ at the weakest setting. "
               f"Predictions are committed as {c['hash12']} ({obs}). {d['filming_instructions']}")
        yield _ev("message", text=txt)
        yield _ev("done", result={"commitment_id": c.get("commitment_id"), "sha256": c.get("sha256")})

    def score(self, commitment_id: str, run_id: str, system_type: str) -> Iterator[dict]:
        yield from self.call("describe_scene", {"run_id": run_id}, "Same setup as the committed experiment, so I'm reusing its scene description and reference object.")
        sc = self.results["describe_scene"]
        yield from self.call("track_objects", {"run_id": run_id}, f"Tracking the {sc['tracked_concepts'][0]} in the new video.")
        yield from self.call("measure", {"run_id": run_id}, "Measuring. For a reveal run, only the first 0.5 s may be used, and only to read off the starting state.")
        yield from self.call("check_validity", {"run_id": run_id}, "Checking this recording is trustworthy and matches the committed setup.")
        yield from self.call("evaluate_commitment", {"commitment_id": commitment_id, "run_id": run_id},
                             "Verifying the commitment hash, then scoring the frozen predictions against what actually happened.")
        v = self.results["evaluate_commitment"]
        yield from self.call("render_ghosts", {"run_id": run_id}, "Rendering the committed predictions as ghosts over the new video.")
        if not v.get("hash_verified", True):
            yield _ev("message", text="Refused: the commitment's hash does not match its contents, so the predictions were altered."); return
        if v.get("validity") == "invalid":
            yield _ev("message", text="INVALID run, so the commitment stays unscored. " + " ".join(v.get("validity_reasons", [])) +
                      " Re-film with the camera fixed and the same setup; the commitment is still open.")
            return
        pm = v["per_model"]
        win, lose = pm[0], pm[-1]
        obs_txt = ""
        if win.get("observable_measured") is not None:
            unit = "s" if win["observable"] == "period_s" else "m/s"
            name = "period" if win["observable"] == "period_s" else "late-fall speed"
            obs_txt = (f" Measured {name}: {win['observable_measured']:.3f} {unit}; {win['model']} predicted "
                       f"{win['observable_predicted']:.3f} ({win['observable_error_pct']:+.1f}%), {lose['model']} predicted "
                       f"{lose['observable_predicted']:.3f} ({lose['observable_error_pct']:+.1f}%).")
        txt = (f"Verdict: {v['winner']}. Hash verified. After the 0.5 s conditioning window, {win['model']} is off by "
               f"{win['rmse_sigma']:.1f}σ RMS while {lose['model']} is off by {lose['rmse_sigma']:.0f}σ.{obs_txt}")
        yield _ev("message", text=txt)


# ---------------- Muse Wonder (Responses API) ----------------
class _MuseWonder:
    def __init__(self):
        self.s = get_settings()
        self.label = f"{self.s.agent_model} via Meta Model API"

    def _create(self, inp: list[dict]) -> dict:
        from openai import OpenAI

        req = {"instructions": PROMPT, "input": inp, "tools": tools.TOOL_DEFS}

        def call():
            client = OpenAI(base_url=self.s.meta_base_url, api_key=self.s.model_api_key)
            r = client.responses.create(model=self.s.agent_model, instructions=PROMPT, input=inp,
                                        tools=tools.TOOL_DEFS, parallel_tool_calls=False)
            return r.model_dump(exclude_none=True)

        return cached_call(self.s.agent_model, req, call)

    def run(self, task_text: str) -> Iterator[dict]:
        inp: list[dict[str, Any]] = [{"role": "user", "content": task_text}]
        n_calls = 0
        while True:
            resp = self._create(inp)
            calls = []
            for item in resp.get("output", []):
                kind = item.get("type")
                if kind == "reasoning":
                    for s in item.get("summary", []) or []:
                        if s.get("text"):
                            yield _ev("thought", text=s["text"])
                elif kind == "message":
                    text = "".join(c.get("text", "") for c in item.get("content", []) if c.get("type") in ("output_text", "text"))
                    if text:
                        yield _ev("thought" if any(i.get("type") == "function_call" for i in resp["output"]) else "message", text=text)
                elif kind == "function_call":
                    calls.append(item)
                inp.append(item)
            if not calls:
                return
            for fc in calls:
                n_calls += 1
                if n_calls > self.s.agent_max_tool_calls:
                    yield _ev("error", text=f"tool-call budget ({self.s.agent_max_tool_calls}) exhausted"); return
                try:
                    args = json.loads(fc.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield _ev("tool_call", id=fc.get("call_id"), name=fc["name"], args=args)
                res = tools.execute(fc["name"], args)
                yield _ev("tool_result", id=fc.get("call_id"), name=fc["name"], ok="error" not in res, result=res)
                inp.append({"type": "function_call_output", "call_id": fc.get("call_id"), "output": json.dumps(res, default=str)})


# ---------------- entry points ----------------
def _context(task: str, ids: dict) -> dict:
    if "run_id" in ids:
        r = storage.load_run(ids["run_id"])
        e = storage.load_experiment(r.experiment_id)
    else:
        e = storage.load_experiment(ids["experiment_id"])
    return {**ids, "experiment_id": e.id, "system_type": e.system_type}


def run_task(task: str, save: bool = True, **ids) -> Iterator[dict]:
    """Run an agent task, yielding notebook events; persists the trace."""
    ctx = _context(task, ids)
    use_llm = get_settings().llm_backend == "meta"
    agent = _MuseWonder() if use_llm else _Mock()
    events: list[dict] = []

    def emit(ev):
        events.append(ev)
        return ev

    yield emit(_ev("status", text=f"agent: {agent.label}", task=task, ids=ctx))
    try:
        if use_llm:
            gen = agent.run(TASKS[task].format(**ctx))
        elif task == "analyze":
            gen = agent.analyze(ctx["run_id"], ctx["experiment_id"], ctx["system_type"])
        elif task == "next":
            gen = agent.next(ctx["experiment_id"], ctx["system_type"])
        else:
            gen = agent.score(ctx["commitment_id"], ctx["run_id"], ctx["system_type"])
        for ev in gen:
            yield emit(ev)
    except Exception as e:  # surface, don't crash the stream
        yield emit(_ev("error", text=f"{type(e).__name__}: {e}"))
    if not any(ev["type"] == "done" for ev in events):
        result = {}
        if task == "next":
            exp = storage.load_experiment(ctx["experiment_id"])
            result = {"commitment_id": exp.commit_ids[-1] if exp.commit_ids else None}
        yield emit(_ev("done", result=result))
    if save:
        key = trace_key(task, ctx)
        storage.save_json(trace_path(key), {"task": task, "ids": ctx, "events": events})
        if "run_id" in ctx:
            r = storage.load_run(ctx["run_id"])
            if task not in r.trace_tasks:
                r.trace_tasks.append(task)
                storage.save_run(r)


def load_trace(task: str, **ids) -> dict | None:
    p = trace_path(trace_key(task, ids))
    return storage.load_json(p) if p.exists() else None


def replay(task: str, delay: float = 0.35, **ids) -> Iterator[dict]:
    tr = load_trace(task, **ids)
    if tr is None:
        raise KeyError(f"no saved trace for {task} {ids}")
    for ev in tr["events"]:
        if delay and ev["type"] in ("tool_call", "message", "thought"):
            time.sleep(delay if ev["type"] != "tool_call" else delay * 1.5)
        yield {**ev, "replay": True}
