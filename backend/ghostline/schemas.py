"""Pydantic data model (SPEC §6)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SystemType = Literal["pendulum", "falling_drag", "spring_mass", "unknown"]


class ReferenceObject(BaseModel):
    concept: str  # e.g. "US quarter"
    known_size_m: float  # e.g. 0.02426; 0 = fixed landmark of unknown size (camera checks only, angles-only measurement)
    dimension: Literal["diameter", "width", "height", "long_side"] = "diameter"


class SceneDescription(BaseModel):
    system_type: str
    tracked_concepts: list[str]
    reference_object: ReferenceObject
    anchor_hint: Optional[str] = None
    gravity_direction_hint: Optional[str] = None
    notes: str = ""
    confidence: float = 0.0


class TrackFrame(BaseModel):
    frame_idx: int
    t: float
    bbox: tuple[float, float, float, float]  # x, y, w, h (px)
    score: float
    polygon: Optional[list[tuple[float, float]]] = None
    mask_rle: Optional[str] = None
    # Backends that see the raw mask may report these directly (sub-pixel).
    centroid: Optional[tuple[float, float]] = None
    area: Optional[float] = None
    min_rect: Optional[tuple[float, float]] = None  # (long side, short side) px


class Track(BaseModel):
    track_id: str
    concept: str
    frames: list[TrackFrame]


class Calibration(BaseModel):
    meters_per_pixel: float
    meters_per_pixel_std: float
    method: Literal["reference_object", "ruler_points", "manual", "angles_only"]
    homography: Optional[list[list[float]]] = None
    pivot_px: Optional[tuple[float, float]] = None
    length_px: Optional[float] = None
    length_m: Optional[float] = None
    circle_fit_rms_px: Optional[float] = None
    gravity_axis: Optional[tuple[float, float]] = None  # unit vector in image coords
    theta_offset: float = 0.0  # equilibrium angle removed from theta (camera roll)
    object_radius_px: Optional[float] = None


class TimeSeries(BaseModel):
    name: str
    unit: str
    t: list[float]
    y: list[float]
    y_smooth: list[float]
    noise_sigma: float
    valid_mask: list[bool]


class ValidityCheck(BaseModel):
    name: str
    status: Literal["pass", "warning", "invalid"]
    value: Optional[float] = None
    threshold: Optional[float] = None
    reason: str


class ValidityReport(BaseModel):
    status: Literal["valid", "warning", "invalid"]
    checks: list[ValidityCheck]

    @property
    def reasons(self) -> list[str]:
        return [c.reason for c in self.checks if c.status != "pass"]


class Split(BaseModel):
    model_config = {"frozen": True}
    train_end_t: float  # holdout = t > train_end_t; never changes after creation
    conditioning_window_s: float = 0.5


class Run(BaseModel):
    id: str
    experiment_id: str
    video_path: str  # normalized mp4, relative to data dir
    original_path: Optional[str] = None
    video_sha256: str
    fps_nominal: float
    width: int = 0
    height: int = 0
    frame_times: list[float]
    keyframes: list[str] = Field(default_factory=list)
    scene: Optional[SceneDescription] = None
    tracks: Optional[dict[str, Track]] = None
    calibration: Optional[Calibration] = None
    series: Optional[dict[str, TimeSeries]] = None
    validity: Optional[ValidityReport] = None
    split: Optional[Split] = None
    knobs: dict[str, float] = Field(default_factory=dict)
    role: Literal["exploration", "reveal"] = "exploration"
    commitment_id: Optional[str] = None
    fits: Optional[list["FitResult"]] = None
    comparison: Optional["Comparison"] = None
    discovery: Optional[dict] = None
    overlay_path: Optional[str] = None
    verdict: Optional["Verdict"] = None
    trace_tasks: list[str] = Field(default_factory=list)
    created_at: str = ""


class Experiment(BaseModel):
    id: str
    name: str
    system_type: SystemType
    run_ids: list[str] = Field(default_factory=list)
    commit_ids: list[str] = Field(default_factory=list)
    description: str = ""
    knob_max: Optional[float] = None  # largest knob value the experimenter's rig allows (e.g. max release angle)
    proposal: Optional["ExperimentProposal"] = None
    created_at: str = ""


class ParamSpec(BaseModel):
    name: str
    unit: str
    lower: float
    upper: float


class ModelSpec(BaseModel):
    name: str
    latex: str
    params: list[ParamSpec]
    state_vars: list[str]
    knob_transfer: str
    observable: str


class FitResult(BaseModel):
    model: str
    run_id: str
    params: dict[str, float]
    param_std: dict[str, float]
    train_rmse: float
    holdout_rmse: Optional[float] = None
    aic: float
    bic: float
    residual_whiteness_p: float
    n_train: int = 0
    n_holdout: int = 0
    band: Optional[dict[str, list[float]]] = None  # bootstrap prediction band {t, lo, hi}


class Comparison(BaseModel):
    fits: list[FitResult]
    best: Optional[str]
    status: Literal["decisive", "ambiguous"]
    max_separation_sigma: float
    explanation_facts: dict


class ExperimentProposal(BaseModel):
    knob: str
    value: float
    expected_separation_sigma: float
    expected_max_separation_sigma: float = 0.0
    discrimination_curve: list[tuple[float, float]]
    feasibility_note: str
    models: list[str] = Field(default_factory=list)
    source_run_id: str = ""
    filming_instructions: str = ""


class CommittedModel(BaseModel):
    name: str
    params: dict[str, float]
    param_std: dict[str, float]
    knob_transfer: str
    predicted_observables: dict[str, float]


class Commitment(BaseModel):
    id: str
    experiment_id: str
    created_at: str
    knob_setting: dict[str, float]
    models: list[CommittedModel]
    protocol: dict
    reference: dict = Field(default_factory=dict)  # setup facts used by validity (e.g. pendulum length)
    canonical_json: str
    sha256: str


class Verdict(BaseModel):
    commitment_id: str
    run_id: str
    hash_verified: bool
    validity: ValidityReport
    per_model: list[dict]
    winner: Optional[str]
    summary_facts: dict


Run.model_rebuild()
Experiment.model_rebuild()
