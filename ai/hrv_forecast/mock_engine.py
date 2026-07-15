"""Deterministic, dependency-free stand-in for the trained personal forecaster.

This is NOT the research model in `HRV /personalized_hrv_system` — it is a
lightweight heuristic (persistence forecast + widening uncertainty bands,
simple EWMA z-score anomaly rule) that implements the *same API contract*
so the rest of Care-Sync can be built and tested against real-shaped
responses today. See model_loader.py for how a real trained checkpoint
takes over automatically once one exists.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone

from .schemas import (
    AnomalyResponse,
    DigitalTwinResponse,
    ForecastResponse,
    HorizonForecast,
    HRVSample,
)

MODEL_VERSION = "mock_persistence_v1"


def _hr_series(samples: list[HRVSample]) -> list[float]:
    return [s.hr for s in samples]


def _rmssd_series(samples: list[HRVSample]) -> list[float]:
    return [s.rmssd for s in samples if s.rmssd is not None]


def _vital_series(samples: list[HRVSample], attr: str) -> list[float]:
    return [getattr(s, attr) for s in samples if getattr(s, attr) is not None]


def forecast(subject_id: str, samples: list[HRVSample], horizons_s: list[int]) -> ForecastResponse:
    hr = _hr_series(samples)
    last_hr = hr[-1]
    hr_std = statistics.pstdev(hr) if len(hr) > 1 else max(1.0, last_hr * 0.03)
    hr_std = max(hr_std, 1.0)

    rmssd_vals = _rmssd_series(samples)
    last_rmssd = rmssd_vals[-1] if rmssd_vals else None
    rmssd_std = statistics.pstdev(rmssd_vals) if len(rmssd_vals) > 1 else (max(1.0, last_rmssd * 0.1) if last_rmssd else None)

    # TEMP/EDA: simple last-value persistence (no interval — these are
    # secondary vitals, shown as a point estimate only). Only populated when
    # the caller actually sent temp/eda samples.
    temp_vals = _vital_series(samples, "temp")
    last_temp = temp_vals[-1] if temp_vals else None
    eda_vals = _vital_series(samples, "eda")
    last_eda = eda_vals[-1] if eda_vals else None

    horizons = []
    for h in sorted(horizons_s):
        # Uncertainty widens with the sqrt of the horizon, like a random walk.
        widen = math.sqrt(h / 60.0)
        hr_sigma = hr_std * widen
        horizons.append(
            HorizonForecast(
                horizon_s=h,
                hr_pred=round(last_hr, 2),
                hr_lower=round(last_hr - 1.96 * hr_sigma, 2),
                hr_upper=round(last_hr + 1.96 * hr_sigma, 2),
                rmssd_pred=round(last_rmssd, 2) if last_rmssd is not None else None,
                rmssd_lower=round(last_rmssd - 1.96 * rmssd_std, 2) if last_rmssd is not None else None,
                rmssd_upper=round(last_rmssd + 1.96 * rmssd_std, 2) if last_rmssd is not None else None,
                temp_pred=round(last_temp, 2) if last_temp is not None else None,
                eda_pred=round(last_eda, 3) if last_eda is not None else None,
            )
        )

    return ForecastResponse(
        subject_id=subject_id,
        generated_at=datetime.now(timezone.utc),
        model_status="mock",
        model_version=MODEL_VERSION,
        horizons=horizons,
    )


def score_anomaly(subject_id: str, samples: list[HRVSample]) -> AnomalyResponse:
    hr = _hr_series(samples)
    mean_hr = statistics.fmean(hr)
    std_hr = statistics.pstdev(hr) if len(hr) > 1 else 1.0
    std_hr = max(std_hr, 1.0)
    last_hr = hr[-1]
    hr_z = (last_hr - mean_hr) / std_hr

    rmssd_vals = _rmssd_series(samples)
    rmssd_pct_drop = 0.0
    if len(rmssd_vals) > 1:
        baseline_rmssd = statistics.fmean(rmssd_vals[:-1])
        if baseline_rmssd > 0:
            rmssd_pct_drop = max(0.0, (baseline_rmssd - rmssd_vals[-1]) / baseline_rmssd)

    temp_vals = [s.temp for s in samples if s.temp is not None]
    temp_deviation = 0.0
    if len(temp_vals) > 1:
        temp_deviation = max(0.0, temp_vals[-1] - statistics.fmean(temp_vals[:-1]))

    illness_score = max(0.0, (max(hr_z, 0.0) + rmssd_pct_drop * 3 + temp_deviation) / 3)
    combined_score = 0.6 * abs(hr_z) + 0.4 * illness_score

    severity = "normal"
    if combined_score >= 3.0:
        severity = "alert"
    elif combined_score >= 1.5:
        severity = "watch"

    reasons = []
    if hr_z > 1.5:
        reasons.append(f"HR {last_hr:.0f}bpm is {hr_z:.1f} SD above this window's mean")
    if rmssd_pct_drop > 0.2:
        reasons.append(f"RMSSD dropped {rmssd_pct_drop * 100:.0f}% vs. recent baseline")
    if temp_deviation > 0.3:
        reasons.append(f"Temp elevated +{temp_deviation:.1f}C vs. recent baseline")
    if not reasons:
        reasons.append("No significant deviation detected")

    return AnomalyResponse(
        subject_id=subject_id,
        generated_at=datetime.now(timezone.utc),
        model_status="mock",
        is_anomaly=severity != "normal",
        score=round(combined_score, 3),
        severity=severity,
        reasons=reasons,
        components={
            "hr_zscore": round(hr_z, 3),
            "rmssd_pct_drop": round(rmssd_pct_drop, 3),
            "temp_deviation": round(temp_deviation, 3),
            "illness_score": round(illness_score, 3),
        },
    )


def digital_twin(subject_id: str, samples: list[HRVSample]) -> DigitalTwinResponse:
    hr = _hr_series(samples)
    rmssd_vals = _rmssd_series(samples)

    # Without a physio-state classifier we can't split resting/sleep/walking/
    # running, so the mock only fills in an overall resting_hr proxy (5th
    # percentile of the window) and marks itself uncalibrated. The real
    # pipeline (src/personalization/digital_twin.py) fills all four states
    # plus a per-hour circadian table once enough history is available.
    sorted_hr = sorted(hr)
    resting_proxy = sorted_hr[max(0, int(len(sorted_hr) * 0.05) - 1)]

    circadian: dict[str, float] = {}
    buckets: dict[int, list[float]] = {}
    for s in samples:
        buckets.setdefault(s.timestamp.hour, []).append(s.hr)
    for hour, vals in buckets.items():
        circadian[str(hour)] = round(statistics.fmean(vals), 2)

    calibrated = len(samples) >= 3600 * 24  # heuristic: need >=1 day of 1Hz data

    return DigitalTwinResponse(
        subject_id=subject_id,
        generated_at=datetime.now(timezone.utc),
        model_status="mock",
        calibrated=calibrated,
        resting_hr=round(resting_proxy, 2),
        sleep_hr=None,
        walking_hr=None,
        running_hr=None,
        avg_rmssd=round(statistics.fmean(rmssd_vals), 2) if rmssd_vals else None,
        circadian_profile=circadian,
    )
