"""Attempts to load a real trained checkpoint from the HRV research system;
falls back to the mock engine when none is available.

Supports per-user personalization: HRV_MODEL_DIR is treated as a *base*
directory that can contain:

  HRV_MODEL_DIR/global/              population baseline (all users, until
                                      they have a personal model)
  HRV_MODEL_DIR/<subject_id>/         a subject's personally fine-tuned model
  HRV_MODEL_DIR/meta.joblib (etc.)    back-compat: HRV_MODEL_DIR itself is a
                                      single flat checkpoint (old behaviour,
                                      used for every subject)

Resolution order per request: personal dir for this subject_id -> global/ ->
HRV_MODEL_DIR itself (flat). Each resolved directory is loaded once and
cached by directory (so many subjects sharing the global model only pay the
load cost once).

To go live with a real model, see docs/HRV_INTEGRATION.md for the full
`run_pretraining.py` (global) + `run_finetune.py` (per-user) walkthrough.
Restart the ai/ service after training; GET /hrv/status?subject_id=<user>
reports "trained"/"mock" and which checkpoint (personal/global) is serving
that subject. Any load error is logged and the service keeps serving the
mock engine instead of crashing.
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


def _is_checkpoint_dir(d: Path) -> bool:
    return d.is_dir() and (d / "meta.joblib").exists()


def _resolve_checkpoint_dir(subject_id: str) -> tuple[Optional[Path], str]:
    """Returns (path_or_None, source) where source is 'personal', 'global',
    'flat', or 'none' (explains which tier of the resolution order matched)."""
    if not DEFAULT_CHECKPOINT_DIR:
        return None, "none"
    base = Path(DEFAULT_CHECKPOINT_DIR)

    personal = base / subject_id
    if _is_checkpoint_dir(personal):
        return personal, "personal"

    glob = base / "global"
    if _is_checkpoint_dir(glob):
        return glob, "global"

    if _is_checkpoint_dir(base):
        return base, "flat"

    return None, "none"


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

    def is_ready(self) -> bool:
        return True


# Cache keyed by resolved checkpoint directory (str) so multiple subjects
# sharing the global model reuse one loaded instance.
_model_cache: dict[str, Optional[HRVRealModel]] = {}
_load_errors: dict[str, str] = {}


def get_real_model(subject_id: str = "global") -> Optional[HRVRealModel]:
    """Returns the best available loaded HRVRealModel for this subject
    (personal -> global -> flat fallback), or None if nothing usable is
    configured/loadable. Each resolved directory is only attempted once per
    process; results are cached by directory."""
    ckpt_dir, source = _resolve_checkpoint_dir(subject_id)
    if ckpt_dir is None:
        if source == "none" and DEFAULT_CHECKPOINT_DIR:
            _load_errors[subject_id] = f"no checkpoint found for '{subject_id}' under {DEFAULT_CHECKPOINT_DIR}"
        return None

    key = str(ckpt_dir)
    if key in _model_cache:
        return _model_cache[key]

    try:
        model = HRVRealModel(ckpt_dir)
        logger.info("Loaded real HRV model (%s) for subject=%s from %s", source, subject_id, ckpt_dir)
        _model_cache[key] = model
        return model
    except Exception as exc:  # noqa: BLE001 - deliberately broad: never take the API down
        logger.warning("HRV real model failed to load from %s, falling back to mock: %s", ckpt_dir, exc)
        _load_errors[key] = str(exc)
        _model_cache[key] = None
        return None


def status(subject_id: str = "global") -> dict:
    ckpt_dir, source = _resolve_checkpoint_dir(subject_id)
    model = get_real_model(subject_id)
    return {
        "model_status": "trained" if model is not None else "mock",
        "checkpoint_dir": str(ckpt_dir) if ckpt_dir else (DEFAULT_CHECKPOINT_DIR or None),
        "personalized": source == "personal" and model is not None,
        "detail": (
            f"Serving {source} checkpoint from {model.checkpoint_dir} for subject '{subject_id}'"
            if model is not None
            else (
                _load_errors.get(str(ckpt_dir)) or _load_errors.get(subject_id)
                or (f"HRV_MODEL_DIR not set; serving mock_persistence_v1" if not DEFAULT_CHECKPOINT_DIR
                    else f"no usable checkpoint under {DEFAULT_CHECKPOINT_DIR} for subject '{subject_id}'; serving mock_persistence_v1")
            )
        ),
    }
