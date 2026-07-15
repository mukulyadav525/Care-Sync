"""Circadian / time-of-day features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_time_features(grid_index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=grid_index)
    hour = grid_index.hour + grid_index.minute / 60.0
    dow = grid_index.dayofweek

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return out
