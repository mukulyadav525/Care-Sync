"""Skin temperature features: rolling trend and deviation from personal baseline."""
from __future__ import annotations

import pandas as pd

from .hr_features import _rolling_slope


def compute_temp_features(grid: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    temp = grid["TEMP"]
    out = pd.DataFrame(index=grid.index)

    out["TEMP_roll_mean_300s"] = temp.rolling(300, min_periods=30).mean()
    out["TEMP_trend_300s"] = _rolling_slope(temp, 300)

    # personal baseline = long-run expanding mean (proxy for "calibration period" baseline)
    baseline = temp.expanding(min_periods=60).mean()
    out["TEMP_baseline_deviation"] = temp - baseline

    return out
