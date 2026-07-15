"""Facade used by ai/routes/hrv.py.

Anomaly detection and the digital twin run the real HRV feature-engineering
pipeline unconditionally (no trained model needed for those). Forecasting
uses the real trained model when HRV_MODEL_DIR is configured and loads
successfully. In every case, if the real path raises for any reason (not
enough samples yet, a bad checkpoint, an unexpected shape), we log it and
fall back to the deterministic mock engine rather than failing the request
— callers can always tell which one they got via `model_status`.
"""
from __future__ import annotations

import logging

from . import mock_engine, model_loader, real_engine
from .schemas import (
    AnomalyResponse,
    DigitalTwinResponse,
    ForecastResponse,
    HRVSample,
    HRVStatusResponse,
)

logger = logging.getLogger("ai.hrv_forecast")


def get_status() -> HRVStatusResponse:
    s = model_loader.status()
    return HRVStatusResponse(
        model_status=s["model_status"],
        model_version="hrv_personalized_v1" if model_loader.get_real_model() else mock_engine.MODEL_VERSION,
        checkpoint_dir=s["checkpoint_dir"],
        detail=s["detail"],
    )


def forecast(subject_id: str, samples: list[HRVSample], horizons_s: list[int]) -> ForecastResponse:
    try:
        return real_engine.forecast(subject_id, samples, horizons_s)
    except Exception as exc:  # noqa: BLE001 - real path is best-effort, mock always works
        logger.info("hrv forecast falling back to mock for %s: %s", subject_id, exc)
        return mock_engine.forecast(subject_id, samples, horizons_s)


def score_anomaly(subject_id: str, samples: list[HRVSample]) -> AnomalyResponse:
    try:
        return real_engine.score_anomaly(subject_id, samples)
    except Exception as exc:  # noqa: BLE001
        logger.info("hrv anomaly falling back to mock for %s: %s", subject_id, exc)
        return mock_engine.score_anomaly(subject_id, samples)


def digital_twin(subject_id: str, samples: list[HRVSample]) -> DigitalTwinResponse:
    try:
        return real_engine.digital_twin(subject_id, samples)
    except Exception as exc:  # noqa: BLE001
        logger.info("hrv digital-twin falling back to mock for %s: %s", subject_id, exc)
        return mock_engine.digital_twin(subject_id, samples)
