"""Signal-Quality Index (SQI): per-modality and overall data-quality scores.

Real wrist data is full of artifacts (motion, poor contact, saturation). Without
a quality gate, sensor noise gets mistaken for physiological anomalies (e.g. the
F_sensor_noise scenario firing health alerts). This module produces a per-second
quality score in [0, 1] per modality (1 = clean) plus an overall SQI, computed
causally from the feature table. The anomaly layer multiplies its score by the
SQI (or masks alerts) so unreliable windows are flagged as "sensor issue" rather
than "health anomaly".

All checks degrade gracefully: a modality whose columns are absent simply
contributes nothing (its SQI is left at 1.0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Plausible physiological ranges for step-artifact / out-of-range detection.
_TEMP_RANGE = (20.0, 42.0)
_EDA_RANGE = (0.0, 60.0)
_HR_JUMP_BPM = 20.0     # >20 bpm change in 1 s is implausible (artifact)
_TEMP_JUMP_C = 1.0      # >1 C/s skin-temp jump is an artifact
_EDA_JUMP_US = 5.0      # >5 uS/s EDA step is an artifact


def _roll(s: pd.Series, win: int = 30) -> pd.Series:
    """Causal rolling mean of a 0/1 issue flag -> fraction of recent bad seconds."""
    return s.rolling(win, min_periods=1).mean()


def compute_signal_quality(table: pd.DataFrame, window_s: int = 30) -> pd.DataFrame:
    """Return per-second SQI columns (sqi_hr, sqi_hrv, sqi_acc, sqi_eda, sqi_temp,
    sqi_bvp, sqi_overall), each in [0, 1] (1 = good)."""
    idx = table.index
    out = pd.DataFrame(index=idx)

    # ---- HR quality: implausible jumps + interpolated/stale samples ----
    if "HR" in table.columns:
        hr = table["HR"]
        jump = (hr.diff().abs() > _HR_JUMP_BPM).astype(float)
        bad = _roll(jump, window_s)
        if "HR_stale" in table.columns:
            bad = bad + _roll(table["HR_stale"].astype(float), window_s)
        out["sqi_hr"] = (1.0 - bad).clip(0.0, 1.0)
    else:
        out["sqi_hr"] = 1.0

    # ---- HRV quality: directly from the beat-validity fraction (ibi_features) ----
    vf_cols = [c for c in table.columns if c.startswith("RMSSD_valid_fraction_")]
    if vf_cols:
        out["sqi_hrv"] = table[vf_cols].mean(axis=1).clip(0.0, 1.0)
    else:
        out["sqi_hrv"] = 1.0

    # ---- ACC quality: saturation / extreme motion spikes ----
    if "ACC_mag_max" in table.columns:
        amax = table["ACC_mag_max"].astype(float)
        thr = amax.median() + 6.0 * (amax.std() + 1e-6)
        spike = (amax > thr).astype(float)
        out["sqi_acc"] = (1.0 - _roll(spike, window_s)).clip(0.0, 1.0)
    else:
        out["sqi_acc"] = 1.0

    # ---- EDA quality: out-of-range + step artifacts + stale ----
    if "EDA" in table.columns:
        eda = table["EDA"].astype(float)
        oor = ((eda < _EDA_RANGE[0]) | (eda > _EDA_RANGE[1])).astype(float)
        step = (eda.diff().abs() > _EDA_JUMP_US).astype(float)
        bad = _roll(oor + step, window_s)
        if "EDA_stale" in table.columns:
            bad = bad + _roll(table["EDA_stale"].astype(float), window_s)
        out["sqi_eda"] = (1.0 - bad).clip(0.0, 1.0)
    else:
        out["sqi_eda"] = 1.0

    # ---- TEMP quality: out-of-range + impossible jumps + stale ----
    if "TEMP" in table.columns:
        temp = table["TEMP"].astype(float)
        oor = ((temp < _TEMP_RANGE[0]) | (temp > _TEMP_RANGE[1])).astype(float)
        step = (temp.diff().abs() > _TEMP_JUMP_C).astype(float)
        bad = _roll(oor + step, window_s)
        if "TEMP_stale" in table.columns:
            bad = bad + _roll(table["TEMP_stale"].astype(float), window_s)
        out["sqi_temp"] = (1.0 - bad).clip(0.0, 1.0)
    else:
        out["sqi_temp"] = 1.0

    # ---- BVP quality: flatline / amplitude collapse (if a BVP amplitude exists) ----
    bvp_amp = next((c for c in table.columns if c.lower().startswith("bvp") and
                    ("amp" in c.lower() or "std" in c.lower() or "energy" in c.lower())), None)
    if bvp_amp is not None:
        amp = table[bvp_amp].astype(float)
        floor = max(amp.median() * 0.1, 1e-6)
        flat = (amp < floor).astype(float)
        out["sqi_bvp"] = (1.0 - _roll(flat, window_s)).clip(0.0, 1.0)
    else:
        out["sqi_bvp"] = 1.0

    # ---- overall = the weakest modality (a single bad sensor ruins the window) ----
    sub = ["sqi_hr", "sqi_hrv", "sqi_acc", "sqi_eda", "sqi_temp", "sqi_bvp"]
    out["sqi_overall"] = out[sub].min(axis=1)
    return out


def quality_label(sqi_overall: np.ndarray, good: float = 0.8, poor: float = 0.5) -> np.ndarray:
    """Map overall SQI to {'good', 'noisy', 'artifact'}."""
    sqi = np.asarray(sqi_overall, float)
    lab = np.full(sqi.shape, "good", dtype=object)
    lab[sqi < good] = "noisy"
    lab[sqi < poor] = "artifact"
    return lab


def quality_downweight(sqi_overall: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Multiplier in [floor, 1] for the anomaly score: poor quality -> downweighted.
    Anomaly_score_effective = anomaly_score * quality_downweight(sqi)."""
    return np.clip(np.asarray(sqi_overall, float), floor, 1.0)
