"""Local filesystem JSON storage: data/{experiments,runs,commits,cache}."""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .config import get_settings
from .schemas import Commitment, Experiment, Run

T = TypeVar("T", bound=BaseModel)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


def data_dir() -> Path:
    d = get_settings().data_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dir(kind: str) -> Path:
    p = data_dir() / kind
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(run_id: str) -> Path:
    p = _dir("runs") / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def abs_path(rel: str) -> Path:
    return data_dir() / rel


def rel_path(p: Path) -> str:
    return str(Path(p).resolve().relative_to(data_dir().resolve()))


def _write(path: Path, model: BaseModel) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(model.model_dump_json(indent=1))
    tmp.replace(path)


# ---- experiments ----
def save_experiment(e: Experiment) -> Experiment:
    _write(_dir("experiments") / f"{e.id}.json", e)
    return e


def load_experiment(eid: str) -> Experiment:
    p = _dir("experiments") / f"{eid}.json"
    if not p.exists():
        raise KeyError(f"experiment {eid} not found")
    return Experiment.model_validate_json(p.read_text())


def list_experiments() -> list[Experiment]:
    out = [Experiment.model_validate_json(p.read_text()) for p in sorted(_dir("experiments").glob("*.json"))]
    return sorted(out, key=lambda e: e.created_at)


# ---- runs ----
def save_run(r: Run) -> Run:
    _write(run_dir(r.id) / "run.json", r)
    return r


def load_run(rid: str) -> Run:
    p = _dir("runs") / rid / "run.json"
    if not p.exists():
        raise KeyError(f"run {rid} not found")
    return Run.model_validate_json(p.read_text())


# ---- commitments: write-once ----
class ImmutableError(RuntimeError):
    pass


def save_commitment(c: Commitment) -> Commitment:
    p = _dir("commits") / f"{c.id}.json"
    if p.exists():
        raise ImmutableError(f"commitment {c.id} already exists and is immutable")
    p.write_text(c.model_dump_json(indent=1))
    p.chmod(0o444)
    return c


def load_commitment(cid: str) -> Commitment:
    p = _dir("commits") / f"{cid}.json"
    if not p.exists():
        raise KeyError(f"commitment {cid} not found")
    return Commitment.model_validate_json(p.read_text())


# ---- arbitrary JSON blobs (agent traces etc.) ----
def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def load_json(path: Path):
    return json.loads(path.read_text())
