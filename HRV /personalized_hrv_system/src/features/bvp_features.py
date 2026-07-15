"""BVP (PPG) features: pulse peaks, amplitude, and pulse-rate variability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _infer_fs(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 64.0
    return 1.0 / (index[1] - index[0]).total_seconds()


def compute_bvp_features(grid_index: pd.DatetimeIndex, bvp_clean: pd.Series, cfg: dict) -> pd.DataFrame:
    fs = _infer_fs(bvp_clean.index)
    min_distance = max(1, int(fs * 60.0 / 220.0))  # cap pulse rate at 220 bpm

    values = bvp_clean.to_numpy()
    peaks, props = find_peaks(values, distance=min_distance, prominence=0.0)

    out = pd.DataFrame(index=grid_index)

    if len(peaks) == 0:
        out["BVP_pulse_rate_1min"] = np.nan
        out["BVP_amplitude_mean_30s"] = np.nan
        out["BVP_amplitude_std_30s"] = np.nan
        return out

    peak_times = bvp_clean.index[peaks]
    amplitudes = pd.Series(props["prominences"], index=peak_times, name="amplitude")

    # pulse count per second -> rolling pulse rate (beats/min)
    counts = amplitudes.resample("1s").count()
    counts = counts.reindex(grid_index, fill_value=0)
    out["BVP_pulse_rate_1min"] = counts.rolling(60, min_periods=10).sum()

    amp_per_sec = amplitudes.resample("1s").mean().reindex(grid_index)
    out["BVP_amplitude_mean_30s"] = amp_per_sec.rolling(30, min_periods=5).mean()
    out["BVP_amplitude_std_30s"] = amp_per_sec.rolling(30, min_periods=5).std()

    return out
