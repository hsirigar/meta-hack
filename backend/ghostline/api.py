"""FastAPI routes (SPEC §19). Agent steps stream over SSE as the lab notebook."""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import pipeline as P
from . import storage
from .agent import loop as agent
from .config import REPO_ROOT, get_settings
from .scene import scene_backend_label
from .schemas import SceneDescription
from .segment import track_summary

app = FastAPI(title="Ghostline", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _j(model) -> dict:
    return json.loads(model.model_dump_json())


def _404(fn: Callable, *a):
    try:
        return fn(*a)
    except KeyError as e:
        raise HTTPException(404, str(e))


# ---------- SSE helper ----------
def _sse(gen_factory: Callable[[], Iterator[dict]]) -> EventSourceResponse:
    async def stream():
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker():
            try:
                for ev in gen_factory():
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            except Exception as e:  # pragma: no cover
                loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "text": f"{type(e).__name__}: {e}"})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        loop.run_in_executor(None, worker)
        while True:
            ev = await q.get()
            if ev is None:
                break
            yield {"event": ev.get("type", "message"), "data": json.dumps(ev, default=str)}

    return EventSourceResponse(stream())


def _agent_stream(task: str, replay: bool, **ids) -> EventSourceResponse:
    want_replay = replay or get_settings().demo_mode
    if want_replay and agent.load_trace(task, **ids) is not None:
        return _sse(lambda: agent.replay(task, **ids))
    return _sse(lambda: agent.run_task(task, **ids))


# ---------- meta ----------
@app.get("/api/health")
def health():
    s = get_settings()
    return {"ok": True, "demo_mode": s.demo_mode, "scene_backend": scene_backend_label(),
            "segment_backend": s.segment_backend, "llm_backend": s.llm_backend,
            "agent_model": s.agent_model if s.llm_backend == "meta" else "scripted mock",
            "muse_spark_open": s.muse_spark_open_active}


# ---------- experiments ----------
class NewExperiment(BaseModel):
    name: str
    system_type: str
    description: str = ""


def _run_summary(r) -> dict:
    return {"id": r.id, "role": r.role, "knobs": r.knobs, "created_at": r.created_at,
            "commitment_id": r.commitment_id,
            "validity": r.validity.status if r.validity else None,
            "comparison": {"status": r.comparison.status, "best": r.comparison.best,
                           "max_separation_sigma": r.comparison.max_separation_sigma} if r.comparison else None,
            "verdict": {"winner": r.verdict.winner, "hash_verified": r.verdict.hash_verified} if r.verdict else None,
            "has_overlay": bool(r.overlay_path), "trace_tasks": r.trace_tasks}


@app.get("/api/experiments")
def list_experiments():
    out = []
    for e in storage.list_experiments():
        d = _j(e)
        d["runs"] = [_run_summary(storage.load_run(r)) for r in e.run_ids]
        out.append(d)
    return out


@app.post("/api/experiments")
def create_experiment(body: NewExperiment):
    if body.system_type not in ("pendulum", "falling_drag"):
        raise HTTPException(400, "system_type must be pendulum or falling_drag")
    return _j(P.create_experiment(body.name, body.system_type, body.description))


@app.get("/api/experiments/{eid}")
def get_experiment(eid: str):
    e = _404(storage.load_experiment, eid)
    d = _j(e)
    d["runs"] = [_run_summary(storage.load_run(r)) for r in e.run_ids]
    d["commitments"] = [_j(storage.load_commitment(c)) for c in e.commit_ids]
    tr = agent.load_trace("next", experiment_id=eid)
    d["next_trace"] = tr["events"] if tr else None
    return d


async def _save_upload(video: UploadFile) -> Path:
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    tmp = Path(tempfile.mkdtemp()) / f"upload{suffix}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return tmp


@app.post("/api/experiments/{eid}/runs")
async def upload_run(eid: str, video: UploadFile = File(...), knobs: str = Form("{}")):
    _404(storage.load_experiment, eid)
    tmp = await _save_upload(video)
    try:
        k = {kk: float(v) for kk, v in json.loads(knobs or "{}").items()}
        r = await asyncio.to_thread(P.add_run, eid, tmp, k)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    return _run_summary(r)


@app.post("/api/experiments/{eid}/next")
def next_experiment(eid: str, replay: bool = False):
    _404(storage.load_experiment, eid)
    return _agent_stream("next", replay, experiment_id=eid)


class DesignBody(BaseModel):
    model_names: Optional[list[str]] = None
    knob: Optional[str] = None


@app.post("/api/experiments/{eid}/design")
def design(eid: str, body: DesignBody = DesignBody()):
    try:
        return _j(P.design(eid, body.model_names, body.knob))
    except ValueError as e:
        raise HTTPException(400, str(e))


class CommitBody(BaseModel):
    model_names: Optional[list[str]] = None
    knob_setting: Optional[dict[str, float]] = None


@app.post("/api/experiments/{eid}/commit")
def commit(eid: str, body: CommitBody = CommitBody()):
    try:
        return _j(P.commit(eid, body.model_names, body.knob_setting))
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------- runs ----------
@app.get("/api/runs/{rid}")
def get_run(rid: str):
    r = _404(storage.load_run, rid)
    d = json.loads(r.model_dump_json(exclude={"tracks"}))
    d["tracks"] = track_summary(r.tracks) if r.tracks else None
    d["curves"] = P.load_curves(rid)
    traces = {}
    for task in r.trace_tasks:
        tr = agent.load_trace(task, run_id=rid)
        if tr:
            traces[task] = tr["events"]
    d["traces"] = traces
    return d


@app.put("/api/runs/{rid}/scene")
def put_scene(rid: str, scene: SceneDescription):
    _404(storage.load_run, rid)
    return _j(P.set_scene(rid, scene).scene)


@app.post("/api/runs/{rid}/analyze")
def analyze(rid: str, replay: bool = False):
    _404(storage.load_run, rid)
    return _agent_stream("analyze", replay, run_id=rid)


@app.post("/api/runs/{rid}/score")
def score(rid: str, replay: bool = False):
    r = _404(storage.load_run, rid)
    if not r.commitment_id:
        raise HTTPException(400, "run is not a reveal run")
    return _agent_stream("score", replay, run_id=rid, commitment_id=r.commitment_id)


class HomographyBody(BaseModel):
    image_points: list[tuple[float, float]]  # 4 clicked pixel points
    world_points: list[tuple[float, float]]  # same 4 points in metres on the plane of motion


@app.post("/api/runs/{rid}/homography")
def set_homography(rid: str, body: HomographyBody):
    if len(body.image_points) != 4 or len(body.world_points) != 4:
        raise HTTPException(400, "need exactly 4 image points and 4 world points")
    from .schemas import Calibration
    r = _404(storage.load_run, rid)
    H = cv2.getPerspectiveTransform(np.float32(body.image_points), np.float32(body.world_points))
    r.calibration = Calibration(meters_per_pixel=1.0, meters_per_pixel_std=0.0, method="ruler_points", homography=H.tolist())
    storage.save_run(r)
    return {"homography": H.tolist(), "note": "re-run measure to apply"}


def _file(rel: Optional[str], media: str):
    if not rel:
        raise HTTPException(404, "not rendered yet")
    p = storage.abs_path(rel)
    if not p.exists():
        raise HTTPException(404, "file missing")
    return FileResponse(p, media_type=media)


@app.get("/api/runs/{rid}/overlay.mp4")
def overlay(rid: str):
    return _file(_404(storage.load_run, rid).overlay_path, "video/mp4")


@app.get("/api/runs/{rid}/video.mp4")
def video(rid: str):
    return _file(_404(storage.load_run, rid).video_path, "video/mp4")


@app.get("/api/runs/{rid}/keyframe/{i}")
def keyframe(rid: str, i: int):
    r = _404(storage.load_run, rid)
    if i < 0 or i >= len(r.keyframes):
        raise HTTPException(404, "no such keyframe")
    return _file(r.keyframes[i], "image/jpeg")


# ---------- commitments ----------
@app.get("/api/commits/{cid}")
def get_commit(cid: str):
    from .commit import verify_commitment
    c = _404(storage.load_commitment, cid)
    return {**_j(c), "verified": verify_commitment(c)}


@app.post("/api/commits/{cid}/reveal")
async def reveal(cid: str, video: UploadFile = File(...), knobs: str = Form("{}")):
    c = _404(storage.load_commitment, cid)
    tmp = await _save_upload(video)
    try:
        k = {kk: float(v) for kk, v in json.loads(knobs or "{}").items()} or dict(c.knob_setting)
        r = await asyncio.to_thread(P.add_run, c.experiment_id, tmp, k, "reveal", cid)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    return _agent_stream("score", False, run_id=r.id, commitment_id=cid)


@app.get("/api/traces/{task}/{target}")
def get_trace(task: str, target: str):
    p = agent.trace_path(f"{task}_{target}")
    if not p.exists():
        raise HTTPException(404, "no trace")
    return JSONResponse(storage.load_json(p))


# ---------- static frontend (single-service deploy) ----------
_dist = REPO_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = _dist / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(_dist / "index.html")
