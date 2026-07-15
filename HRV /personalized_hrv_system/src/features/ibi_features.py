"""IBI / HRV features: RMSSD, SDNN, pNN50, IBI-derived HR, sampled onto the 1Hz grid.

HRV from wrist PPG is noisy, so before computing RMSSD/SDNN we reject ectopic /
artifact beats at the beat level (Hampel median filter on top of the plausible-
range filter already applied in cleaning.clean_ibi). Each HRV window also carries
a `RMSSD_valid_fraction_{w}s` quality column and requires a minimum number of
valid beats, so downstream code can drop or down-weight low-quality HRV targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _hampel_filter(beats: pd.Series, window: int = 5, n_sigma: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Flag ectopic-like beats whose IBI deviates from the local median by more
    than n_sigma robust standard deviations (MAD-based). Returns (cleaned, invalid)
    where cleaned has flagged beats replaced by the local median."""
    med = beats.rolling(window, center=True, min_periods=1).median()
    abs_dev = (beats - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    thresh = n_sigma * 1.4826 * mad
    # where MAD==0 (flat region) treat as valid; only flag clear outliers
    invalid = (thresh > 0) & (abs_dev > thresh)
    cleaned = beats.where(~invalid, med)
    return cleaned, invalid.astype(float)


def compute_ibi_features(grid_index: pd.DatetimeIndex, ibi_clean: pd.Series, cfg: dict) -> pd.DataFrame:
    """Compute trailing-window HRV metrics at each beat and sample onto `grid_index`."""
    windows = cfg["features"]["hrv_windows_s"]
    fcfg = cfg.get("features", {})
    min_beats = fcfg.get("hrv_min_beats", 5)
    hampel_w = fcfg.get("hrv_hampel_window", 5)
    hampel_sigma = fcfg.get("hrv_hampel_sigma", 3.0)
    out = pd.DataFrame(index=grid_index)

    if ibi_clean.empty:
        for w in windows:
            out[f"RMSSD_{w}s"] = np.nan
            out[f"SDNN_{w}s"] = np.nan
            out[f"pNN50_{w}s"] = np.nan
            out[f"HR_from_IBI_{w}s"] = np.nan
            out[f"RMSSD_valid_fraction_{w}s"] = 0.0
        return out

    beats = ibi_clean.sort_index()
    # beat-level artifact rejection (ectopic / motion outliers) before HRV
    beats_clean, invalid = _hampel_filter(beats, window=hampel_w, n_sigma=hampel_sigma)
    valid = 1.0 - invalid
    diffs = beats_clean.diff()
    # successive differences spanning a rejected beat are unreliable -> NaN them
    diffs = diffs.where((valid == 1.0) & (valid.shift(1) == 1.0))

    for w in windows:
        win = f"{w}s"

        rmssd = diffs.rolling(win).apply(
            lambda x: float(np.sqrt(np.nanmean(x ** 2))) if np.isfinite(x).sum() >= min_beats - 1 else np.nan,
            raw=True,
        )
        sdnn = beats_clean.rolling(win).std()
        pnn50 = diffs.rolling(win).apply(
            lambda x: float(np.nanmean(np.abs(x) > 0.05)) if np.isfinite(x).sum() >= min_beats - 1 else np.nan,
            raw=True,
        )
        mean_ibi = beats_clean.rolling(win).mean()
        hr_from_ibi = 60.0 / mean_ibi
        valid_frac = valid.rolling(win).mean()

        # forward-fill each beat-indexed metric onto the 1Hz grid
        out[f"RMSSD_{w}s"] = rmssd.reindex(grid_index, method="ffill")
        out[f"SDNN_{w}s"] = sdnn.reindex(grid_index, method="ffill")
        out[f"pNN50_{w}s"] = pnn50.reindex(grid_index, method="ffill")
        out[f"HR_from_IBI_{w}s"] = hr_from_ibi.reindex(grid_index, method="ffill")
        out[f"RMSSD_valid_fraction_{w}s"] = valid_frac.reindex(grid_index, method="ffill").fillna(0.0)

    return out
