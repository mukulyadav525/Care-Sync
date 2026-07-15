"""HR-derived features: rolling mean/std, trend, rate of change."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """Slope (per second) of a linear fit of `s` over a trailing window."""
    if window < 2:
        return pd.Series(0.0, index=s.index)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < 1e-10:
        return pd.Series(0.0, index=s.index)

    def slope(y: np.ndarray) -> float:
        return float(((x - x_mean) * (y - y.mean())).sum() / denom)

    return s.rolling(window, min_periods=window).apply(slope, raw=True)


def compute_hr_features(grid: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Given the synced 1Hz grid (must contain 'HR'), compute HR-derived features."""
    freq_s = 1  # grid is 1Hz
    windows = cfg["features"]["hr_windows_s"]
    hr = grid["HR"]

    out = pd.DataFrame(index=grid.index)
    for w in windows:
        out[f"HR_roll_mean_{w}s"] = hr.rolling(w, min_periods=max(1, w // 2)).mean()
        out[f"HR_roll_std_{w}s"] = hr.rolling(w, min_periods=max(1, w // 2)).std()
        out[f"HR_trend_{w}s"] = _rolling_slope(hr, w)

    for delta in (5, 30, 60):
        out[f"HR_roc_{delta}s"] = hr - hr.shift(delta)

    return out
