"""Real (non-mock) implementations backed by the HRV research pipeline.

Anomaly detection and the digital twin only need feature engineering — no
trained neural net — so they run for real today, no checkpoint required.
Forecasting does need a trained model; forecast() raises if none is loaded
so the caller (service.py) can fall back to the mock engine.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import feature_adapter, model_loader
from .schemas import (
    AnomalyResponse,
    DigitalTwinResponse,
    ForecastResponse,
    HorizonForecast,
    HRVSample,
)


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


_RMSSD_SIGNAL_RE = re.compile(r"^RMSSD_(\d+s|provided)$")


def _rmssd_col(table: pd.DataFrame) -> str | None:
    """Picks the RMSSD signal column with the most real data — strictly the
    raw RMSSD_{w}s columns (from IBI beats) or RMSSD_provided (from
    caller-supplied `rmssd` values, added by feature_adapter.build_table).
    Deliberately excludes derived columns like RMSSD_minus_baseline_* /
    RMSSD_z_rolling_* which default to 0.0 rather than NaN and would
    otherwise look "more complete" than a genuinely all-NaN signal column."""
    candidates = [c for c in table.columns if _RMSSD_SIGNAL_RE.match(c)]
    candidates = [c for c in candidates if table[c].notna().any()]
    if not candidates:
        return None
    return max(candidates, key=lambda c: table[c].notna().sum())


def digital_twin(subject_id: str, samples: list[HRVSample]) -> DigitalTwinResponse:
    from src.personalization.digital_twin import build_digital_twin  # noqa: E402

    table = feature_adapter.build_table(samples, subject_id)
    twin = build_digital_twin(table, subject_id)

    avg_rmssd = twin.get("avg_rmssd")
    if avg_rmssd is None or (isinstance(avg_rmssd, float) and math.isnan(avg_rmssd)):
        rcol = _rmssd_col(table)
        avg_rmssd = float(table[rcol].dropna().mean()) if rcol is not None else None
    twin["avg_rmssd"] = avg_rmssd

    circadian_profile = {
        hour: _safe_float(entry.get("hr_mean"), default=None) if entry.get("hr_mean") is not None else None
        for hour, entry in twin["circadian_profile"].items()
        if entry.get("hr_mean") is not None
    }
    calibrated = twin["resting_hr"] is not None and len(samples) >= 3600  # >=1h of data

    return DigitalTwinResponse(
        subject_id=subject_id,
        generated_at=datetime.now(timezone.utc),
        model_status="pipeline",
        calibrated=calibrated,
        resting_hr=twin["resting_hr"],
        sleep_hr=twin["sleep_hr"],
        walking_hr=twin["walking_hr"],
        running_hr=twin["running_hr"],
        avg_rmssd=twin["avg_rmssd"],
        circadian_profile={k: v for k, v in circadian_profile.items() if v is not None},
    )


def score_anomaly(subject_id: str, samples: list[HRVSample]) -> AnomalyResponse:
    from src.anomaly.explain import build_alert_reason, illness_score
    from src.personalization.digital_twin import build_digital_twin, digital_twin_score, expected_hr

    table = feature_adapter.build_table(samples, subject_id)
    twin = build_digital_twin(table, subject_id)
    last = table.iloc[-1]
    last_ts = table.index[-1]

    hr_actual = _safe_float(last.get("HR"))
    physio_state = int(last["physio_state"]) if "physio_state" in table.columns and not pd.isna(last.get("physio_state")) else None
    physio_state_name = last.get("physio_state_name") if "physio_state_name" in table.columns else None

    circadian_z = last.get("HR_circadian_zscore")
    circadian_z = _safe_float(circadian_z, default=None) if circadian_z is not None and not pd.isna(circadian_z) else None

    expected, _std = expected_hr(twin, int(last_ts.hour), physio_state)
    if circadian_z is not None:
        hr_z = circadian_z
    else:
        hr_z = digital_twin_score(hr_actual, int(last_ts.hour), twin, physio_state)
    expected_val = expected if expected is not None else hr_actual

    rcol = _rmssd_col(table)
    rmssd_pct_drop = 0.0
    if rcol is not None:
        series = table[rcol].dropna()
        if len(series) > 1:
            baseline = float(series.iloc[:-1].mean())
            latest = _safe_float(series.iloc[-1], default=baseline)
            if baseline > 0:
                rmssd_pct_drop = max(0.0, (baseline - latest) / baseline)

    temp_deviation = 0.0
    temp_baseline_col = "TEMP_minus_baseline_3600s"
    if temp_baseline_col in table.columns:
        temp_deviation = max(0.0, _safe_float(last.get(temp_baseline_col)))

    illness = float(illness_score(np.array([hr_z]), np.array([rmssd_pct_drop]), np.array([temp_deviation]))[0])
    combined_score = 0.6 * abs(hr_z) + 0.4 * illness

    severity = "normal"
    if combined_score >= 3.0:
        severity = "alert"
    elif combined_score >= 1.5:
        severity = "watch"

    reason = build_alert_reason(
        hr_actual=hr_actual,
        hr_expected=expected_val,
        rmssd_pct_drop=rmssd_pct_drop,
        temp_deviation=temp_deviation,
        activity_state_name=physio_state_name,
        circadian_z=circadian_z,
    )

    return AnomalyResponse(
        subject_id=subject_id,
        generated_at=datetime.now(timezone.utc),
        model_status="pipeline",
        is_anomaly=severity != "normal",
        score=round(combined_score, 3),
        severity=severity,
        reasons=[reason],
        components={
            "hr_zscore": round(hr_z, 3),
            "rmssd_pct_drop": round(rmssd_pct_drop, 3),
            "temp_deviation": round(temp_deviation, 3),
            "illness_score": round(illness, 3),
        },
    )


def forecast(subject_id: str, samples: list[HRVSample], horizons_s: list[int]) -> ForecastResponse:
    """Real ML forecast. Raises if no trained checkpoint is loaded, or if the
    feature/window requirements aren't met yet — callers should catch and
    fall back to the mock engine."""
    real_model = model_loader.get_real_model(subject_id)
    if real_model is None:
        raise model_loader.RealModelUnavailable("no trained checkpoint loaded (set HRV_MODEL_DIR)")
    personalized = subject_id in str(real_model.checkpoint_dir)

    from src.models import datasets

    table = feature_adapter.build_table(samples, subject_id)
    meta = real_model.meta
    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]

    missing = [c for c in feature_cols if c not in table.columns]
    for c in missing:
        table[c] = 0.0
    if len(table) < seq_len:
        raise feature_adapter.InsufficientData(f"need >= {seq_len}s of samples for this model, got {len(table)}")

    window = table[feature_cols].iloc[-seq_len:].to_numpy(dtype=float)
    window = np.nan_to_num(window, nan=0.0)
    scaled = real_model.scaler.transform(window)

    model_type = meta["model_type"]
    if model_type == "xgboost":
        preds = real_model.model.predict(scaled.reshape(1, -1))
    else:
        import torch

        with torch.no_grad():
            x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
            out = real_model.model(x)
            preds = out[0].numpy() if isinstance(out, (tuple, list)) else out.numpy()
    preds = np.asarray(preds).reshape(-1)

    target_scaler = meta.get("_target_scaler")
    if target_scaler is not None:
        preds = target_scaler.inverse_transform(preds.reshape(1, -1))[0]

    trained_horizons = meta.get("horizons_s") or DEFAULT_HORIZONS_FALLBACK
    n_horizons = len(trained_horizons)
    hr_preds = preds[:n_horizons]
    rmssd_preds = preds[n_horizons: 2 * n_horizons] if meta.get("predict_rmssd") else [None] * n_horizons

    horizons = []
    for h, hr_p, rmssd_p in zip(trained_horizons, hr_preds, rmssd_preds):
        if h not in horizons_s:
            continue
        sigma = max(1.0, abs(float(hr_p)) * 0.05)
        horizons.append(
            HorizonForecast(
                horizon_s=h,
                hr_pred=round(float(hr_p), 2),
                hr_lower=round(float(hr_p) - 1.96 * sigma, 2),
                hr_upper=round(float(hr_p) + 1.96 * sigma, 2),
                rmssd_pred=round(float(rmssd_p), 2) if rmssd_p is not None else None,
                rmssd_lower=None,
                rmssd_upper=None,
            )
        )
    if not horizons:
        raise feature_adapter.InsufficientData("model's trained horizons don't overlap requested horizons_s")

    return ForecastResponse(
        subject_id=subject_id,
        generated_at=datetime.now(timezone.utc),
        model_status="trained",
        model_version=f"hrv_{model_type}_v1",
        personalized=personalized,
        horizons=horizons,
    )


DEFAULT_HORIZONS_FALLBACK = [60, 300, 600]
