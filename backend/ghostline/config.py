"""Runtime configuration (env / .env at repo root)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    # Meta Model API
    model_api_key: str = ""
    meta_base_url: str = "https://api.meta.ai/v1"
    agent_model: str = "muse-wonder-1.0"
    vision_model: str = "muse-spark-1.3"
    sam_model: str = "sam-3.1"

    # Baseten (Muse Spark Open)
    baseten_api_key: str = ""
    baseten_base_url: str = ""
    baseten_model: str = "meta-models/muse-spark-open-1026"

    # Backend selection
    scene_backend: str = "mock"  # mock | meta | baseten
    segment_backend: str = "fallback"  # fallback | meta | local
    llm_backend: str = "mock"  # mock | meta

    demo_mode: bool = False
    data_dir: Path = REPO_ROOT / "data"

    # Pipeline thresholds
    sam_min_score: float = 0.5
    train_fraction: float = 0.6
    conditioning_window_s: float = 0.5

    # Ambiguity rule (SPEC §10)
    ambig_dbic: float = 6.0
    ambig_rmse_ratio: float = 1.2
    ambig_sep_sigma: float = 3.0

    # Validity (SPEC §7.5)
    camera_drift_frac: float = 0.03
    camera_jump_sigma: float = 10.0
    scale_change_frac: float = 0.05
    tracking_lost_invalid: float = 0.15
    tracking_lost_warning: float = 0.05
    timing_gap_factor: float = 3.0
    setup_mismatch_frac: float = 0.05

    agent_max_tool_calls: int = 25

    @property
    def muse_spark_open_active(self) -> bool:
        return self.scene_backend == "baseten"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if not s.data_dir.is_absolute():
        s.data_dir = (REPO_ROOT / "backend" / s.data_dir).resolve()
    return s
