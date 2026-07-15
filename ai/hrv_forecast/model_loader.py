"""Attempts to load a real trained checkpoint from the HRV research system;
falls back to the mock engine when none is available (which is the case
today — see docs/HRV_INTEGRATION.md).

To go live with a real model:

  1. Train a personal or population model with the pipeline in
     `HRV /personalized_hrv_system` (e.g. `python scripts/run_pretraining.py`
     or `python scripts/run_training.py`, see that folder's DESIGN.md).
  2. Point HRV_MODEL_DIR at the resulting output directory (the one
     containing meta.joblib, scaler.joblib, and the `*_model.pt` /
     `xgb/` files written by src/models/train.py).
  3. Install the extra deps those modules need: torch, joblib, xgboost,
     scikit-learn (see `HRV /personalized_hrv_system/requirements.txt`).
  4. Restart the ai/ service. GET /hrv/status will report "trained" once
     the checkpoint loads successfully; any load error is logged and the
     service keeps serving the mock engine instead of crashing.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env here too (not just in ai/config.py) since this module reads
# HRV_MODEL_DIR from the environment at import time, and import order
# elsewhere in the app isn't guaranteed to load .env first.
load_dotenv()

logger = logging.getLogger("ai.hrv_forecast")

# ai/hrv_forecast/model_loader.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HRV_SYSTEM_ROOT = _REPO_ROOT / "HRV " / "personalized_hrv_system"

DEFAULT_CHECKPOINT_DIR = os.environ.get("HRV_MODEL_DIR", "")


class RealModelUnavailable(Exception):
    """Raised internally when no usable checkpoint/deps are present."""


class HRVRealModel:
    """Thin wrapper around HRV/personalized_hrv_system's inference pipeline."""

    def __init__(self, checkpoint_dir: Path):
        if str(_HRV_SYSTEM_ROOT) not in sys.path:
            sys.path.insert(0, str(_HRV_SYSTEM_ROOT))
        try:
            from src.pipeline import inference_pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover - only hit without torch/joblib etc.
            raise RealModelUnavailable(f"could not import HRV inference pipeline: {exc}") from exc

        self.model, self.scaler, self.meta = inference_pipeline.load_model(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

    # NOTE: run_inference() in the research pipeline expects a fully feature-
    # engineered pandas table (see src/features/build_features.py), not raw
    # HRVSample objects. Wiring that feature-engineering step in front of
    # this call is the remaining step for full production inference and is
    # intentionally left as a TODO so this loader can ship without pulling
    # torch/xgboost into the default ai/ install.
    def is_ready(self) -> bool:
        return True


_cached_model: Optional[HRVRealModel] = None
_load_attempted = False
_load_error: Optional[str] = None


def get_real_model() -> Optional[HRVRealModel]:
    """Returns a loaded HRVRealModel, or None if unavailable. Only tries once
    per process; the result is cached."""
    global _cached_model, _load_attempted, _load_error

    if _load_attempted:
        return _cached_model

    _load_attempted = True
    if not DEFAULT_CHECKPOINT_DIR:
        _load_error = "HRV_MODEL_DIR not set"
        logger.info("HRV real model not loaded: %s", _load_error)
        return None

    ckpt_dir = Path(DEFAULT_CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        _load_error = f"checkpoint dir does not exist: {ckpt_dir}"
        logger.warning("HRV real model not loaded: %s", _load_error)
        return None

    try:
        _cached_model = HRVRealModel(ckpt_dir)
        logger.info("Loaded real HRV model from %s", ckpt_dir)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: never take the API down
        _load_error = str(exc)
        logger.warning("HRV real model failed to load, falling back to mock: %s", exc)
        _cached_model = None

    return _cached_model


def status() -> dict:
    model = get_real_model()
    return {
        "model_status": "trained" if model is not None else "mock",
        "checkpoint_dir": DEFAULT_CHECKPOINT_DIR or None,
        "detail": (
            f"Serving real checkpoint from {model.checkpoint_dir}"
            if model is not None
            else (_load_error or "HRV_MODEL_DIR not set; serving mock_persistence_v1")
        ),
    }
