"""HRV forecasting / anomaly detection / digital twin API.

Backed today by a deterministic mock engine (see ai/hrv_forecast/mock_engine.py);
will transparently switch to the trained model from `HRV /personalized_hrv_system`
once a checkpoint is configured via HRV_MODEL_DIR (see
ai/hrv_forecast/model_loader.py and docs/HRV_INTEGRATION.md). The response
shapes are stable across both, so callers don't need to change when the real
model comes online.
"""
from fastapi import APIRouter, HTTPException

from ai.hrv_forecast import service
from ai.hrv_forecast.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    DigitalTwinRequest,
    DigitalTwinResponse,
    ForecastRequest,
    ForecastResponse,
    HRVStatusResponse,
)

router = APIRouter(prefix="/hrv", tags=["hrv"])


@router.get("/status", response_model=HRVStatusResponse)
def hrv_status():
    """Reports whether requests are being served by the real trained model
    or the mock engine, and why."""
    return service.get_status()


@router.post("/forecast", response_model=ForecastResponse)
def hrv_forecast(req: ForecastRequest):
    try:
        return service.forecast(req.subject_id, req.samples, req.horizons_s)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not compute forecast: {exc}") from exc


@router.post("/anomaly", response_model=AnomalyResponse)
def hrv_anomaly(req: AnomalyRequest):
    try:
        return service.score_anomaly(req.subject_id, req.samples)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not score anomaly: {exc}") from exc


@router.post("/digital-twin", response_model=DigitalTwinResponse)
def hrv_digital_twin(req: DigitalTwinRequest):
    try:
        return service.digital_twin(req.subject_id, req.samples)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not build digital twin: {exc}") from exc
