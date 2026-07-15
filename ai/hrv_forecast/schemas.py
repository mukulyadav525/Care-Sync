"""Pydantic request/response schemas for the /hrv/* API surface.

Field names deliberately mirror the vocabulary used by the research
implementation in `HRV /personalized_hrv_system` (DESIGN.md, src/pipeline,
src/anomaly, src/personalization) so that swapping the mock engine for the
real trained pipeline (see model_loader.py) does not require changing the
API contract that the frontend/backend consume.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class HRVSample(BaseModel):
    """One second (or one beat-window) of wearable data.

    Only `timestamp` and `hr` are required — everything else is optional so
    callers can send whatever the device actually captured (E4-style BVP/EDA/
    TEMP/ACC, or just HR/IBI from a simpler wearable).
    """

    timestamp: datetime
    hr: float = Field(..., description="Instantaneous or 1s-mean heart rate, bpm")
    rmssd: Optional[float] = Field(None, description="RMSSD over the sample window, ms")
    ibi: Optional[float] = Field(None, description="Inter-beat interval, seconds")
    acc_mag: Optional[float] = Field(None, description="Accelerometer magnitude, g")
    temp: Optional[float] = Field(None, description="Skin temperature, C")
    eda: Optional[float] = Field(None, description="Electrodermal activity, microsiemens")


class ForecastRequest(BaseModel):
    subject_id: str
    samples: list[HRVSample] = Field(..., min_length=1)
    horizons_s: list[int] = Field(
        default_factory=lambda: [60, 300, 600],
        description="Forecast horizons in seconds ahead (default 1/5/10 min, matches DESIGN.md)",
    )


class HorizonForecast(BaseModel):
    horizon_s: int
    hr_pred: float
    hr_lower: float
    hr_upper: float
    rmssd_pred: Optional[float] = None
    rmssd_lower: Optional[float] = None
    rmssd_upper: Optional[float] = None
    # Only populated when the caller sent temp/eda samples — persistence
    # forecast in mock mode, multi-task model output once predict_vitals is
    # enabled on a trained checkpoint (see HRV DESIGN.md model.predict_vitals).
    temp_pred: Optional[float] = None
    eda_pred: Optional[float] = None


class ForecastResponse(BaseModel):
    subject_id: str
    generated_at: datetime
    model_status: str = Field(
        ..., description="'trained' if a real checkpoint served this forecast, else 'mock' (persistence heuristic)"
    )
    model_version: str
    horizons: list[HorizonForecast]


class AnomalyRequest(BaseModel):
    subject_id: str
    samples: list[HRVSample] = Field(..., min_length=1)


class AnomalyResponse(BaseModel):
    subject_id: str
    generated_at: datetime
    model_status: str
    is_anomaly: bool
    score: float = Field(..., description="Combined anomaly score (DESIGN.md section 7)")
    # model_status here is 'pipeline' (real feature-engineering, no ML needed)
    # or 'mock' if there wasn't enough data to run the real pipeline.
    severity: str = Field(..., description="'normal' | 'watch' | 'alert'")
    reasons: list[str] = Field(default_factory=list)
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Sub-scores, e.g. hr_zscore, rmssd_pct_drop, temp_deviation, illness_score",
    )


class DigitalTwinRequest(BaseModel):
    subject_id: str
    samples: list[HRVSample] = Field(..., min_length=1)


class DigitalTwinResponse(BaseModel):
    subject_id: str
    generated_at: datetime
    model_status: str
    calibrated: bool = Field(..., description="True once enough history exists for a personal baseline")
    resting_hr: Optional[float] = None
    sleep_hr: Optional[float] = None
    walking_hr: Optional[float] = None
    running_hr: Optional[float] = None
    avg_rmssd: Optional[float] = None
    circadian_profile: dict[str, float] = Field(
        default_factory=dict, description="hour-of-day (0-23, as string) -> expected HR"
    )


class HRVStatusResponse(BaseModel):
    model_status: str
    model_version: str
    checkpoint_dir: Optional[str] = None
    detail: str
