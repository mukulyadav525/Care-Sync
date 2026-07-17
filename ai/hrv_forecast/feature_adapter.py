"""Turns a list of HRVSample into the feature table the HRV research pipeline
(`HRV /personalized_hrv_system/src`) expects, without needing raw E4 files.

This bypasses `src.data.sync.build_synced_frame` (which cleans raw multi-rate
E4 exports) because our input is already a per-second-ish stream from the API
caller. It builds the same "synced" shape (`grid`, `bvp_clean`, `ibi_clean`)
directly, then hands off to `src.features.build_features.build_feature_table`,
which is the real, non-mock feature engineering: HR/HRV rolling features,
circadian baseline, physiological state classification, subject-relative
baselines, and recovery/activity-gating features. None of this requires a
trained model or torch — it's pure numpy/pandas/scipy, already in
ai/requirements.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .schemas import HRVSample

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HRV_SYSTEM_ROOT = _REPO_ROOT / "HRV " / "personalized_hrv_system"

if str(_HRV_SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_HRV_SYSTEM_ROOT))

# Mirrors HRV /personalized_hrv_system/configs/config.yaml's `features`/
# `resample`/`model` sections. Kept as a literal dict (rather than parsing the
# YAML) so ai/ doesn't need a pyyaml dependency just for this.
DEFAULT_CFG = {
    "resample": {"freq": "1s"},
    "features": {
        "hr_windows_s": [30, 60, 300],
        "hrv_windows_s": [60, 300],
        "acc_windows_s": [10, 60],
        "bvp_agg_window_s": 1,
        "activity_thresholds_g": [0.05, 0.2, 0.5],
        "hrv_min_beats": 3,
        "hrv_hampel_window": 5,
        "hrv_hampel_sigma": 3.0,
        "baseline_windows_s": [3600],
        "longctx_windows_s": [900, 1800, 3600],
    },
    "model": {
        "horizons_s": [60, 300, 600],
        "predict_rmssd": True,
        "predict_deltas": False,
        "predict_vitals": False,
    },
}

MIN_SAMPLES_FOR_TABLE = 2


class InsufficientData(Exception):
    pass


def _samples_to_synced(samples: list[HRVSample], subject_id: str, cfg: dict) -> dict:
    freq = cfg["resample"]["freq"]

    idx = pd.DatetimeIndex([s.timestamp for s in samples])
    raw = pd.DataFrame(
        {
            "HR": [s.hr for s in samples],
            "TEMP": [s.temp for s in samples],
            "EDA": [s.eda for s in samples],
            "ACC_mag": [s.acc_mag for s in samples],
        },
        index=idx,
    ).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]

    # Resample first, then build the grid on the resampled (bin-aligned) index.
    # Building the grid on raw.index and assigning the resampled series into it
    # silently produced an all-NaN HR column whenever callers sent sub-second
    # timestamps (the raw index labels never matched the 1s bin labels).
    hr = raw["HR"].resample(freq).mean()
    grid = pd.DataFrame(index=pd.date_range(hr.index.min(), hr.index.max(), freq=freq))
    grid["HR"] = hr.reindex(grid.index).interpolate(limit=5).ffill().bfill()

    for col in ("TEMP", "EDA"):
        if raw[col].notna().any():
            grid[col] = raw[col].resample(freq).mean().reindex(grid.index).interpolate(limit=5)

    if raw["ACC_mag"].notna().any():
        acc = raw["ACC_mag"].resample(freq).agg(["mean", "std", "max"]).reindex(grid.index)
        grid["ACC_mag_mean"] = acc["mean"].interpolate(limit=5)
        grid["ACC_mag_std"] = acc["std"].fillna(0.0)
        grid["ACC_mag_max"] = acc["max"].interpolate(limit=5)

    ibi_pairs = [(s.timestamp, s.ibi) for s in samples if s.ibi is not None]
    if ibi_pairs:
        ibi_idx, ibi_vals = zip(*ibi_pairs)
        ibi_clean = pd.Series(ibi_vals, index=pd.DatetimeIndex(ibi_idx), name="IBI").sort_index()
        ibi_clean = ibi_clean[~ibi_clean.index.duplicated(keep="last")]
    else:
        ibi_clean = pd.Series(dtype=float, name="IBI")

    return {
        "subject_id": subject_id,
        "grid": grid,
        "bvp_clean": pd.Series(dtype=float, name="BVP"),
        "ibi_clean": ibi_clean,
        "tags": pd.DatetimeIndex([]),
    }


def build_table(samples: list[HRVSample], subject_id: str, cfg: dict | None = None) -> pd.DataFrame:
    """Builds the full engineered feature table for a subject's sample window.

    Raises InsufficientData if there aren't enough samples to build a
    meaningful table (callers should fall back to the mock engine).
    """
    if len(samples) < MIN_SAMPLES_FOR_TABLE:
        raise InsufficientData(f"need at least {MIN_SAMPLES_FOR_TABLE} samples, got {len(samples)}")

    cfg = cfg or DEFAULT_CFG
    from src.features import build_features  # noqa: E402  (path set up above)

    synced = _samples_to_synced(samples, subject_id, cfg)
    table = build_features.build_feature_table(synced, cfg)
    if table.empty:
        raise InsufficientData("feature table came back empty")

    # The pipeline's RMSSD_{w}s columns are derived from raw IBI beats
    # (ibi_clean) — if the caller sent a precomputed `rmssd` per sample
    # instead of `ibi`, those columns stay all-NaN. Add a fallback column
    # from the caller-provided values so downstream RMSSD lookups still
    # have something real to use instead of silently going to NaN/None.
    rmssd_pairs = [(s.timestamp, s.rmssd) for s in samples if s.rmssd is not None]
    if rmssd_pairs:
        idx, vals = zip(*rmssd_pairs)
        provided = pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float).sort_index()
        provided = provided[~provided.index.duplicated(keep="last")]
        table["RMSSD_provided"] = provided.reindex(table.index).interpolate(limit=60).ffill().bfill()

    return table
