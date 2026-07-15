"""Accelerometer-derived features: activity intensity, movement flag, activity bucket."""
from __future__ import annotations

import numpy as np
import pandas as pd


ACTIVITY_LABELS = ["REST", "LIGHT", "MODERATE", "VIGOROUS"]


def compute_acc_features(grid: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Requires grid columns ACC_mag_mean/std/max (gravity-removed magnitude, in g)."""
    windows = cfg["features"]["acc_windows_s"]
    thresholds = cfg["features"]["activity_thresholds_g"]
    mag = grid["ACC_mag_mean"].abs()

    out = pd.DataFrame(index=grid.index)
    for w in windows:
        out[f"ACC_intensity_{w}s"] = mag.rolling(w, min_periods=max(1, w // 2)).mean()
        out[f"ACC_intensity_std_{w}s"] = mag.rolling(w, min_periods=max(1, w // 2)).std()

    primary = out[f"ACC_intensity_{windows[0]}s"]
    out["movement_flag"] = (primary > thresholds[0]).astype(int)

    bins = [-np.inf, *thresholds, np.inf]
    bucket_idx = pd.cut(primary, bins=bins, labels=False, include_lowest=True)
    out["activity_bucket"] = bucket_idx
    out["activity_bucket_name"] = bucket_idx.map(lambda i: ACTIVITY_LABELS[int(i)] if pd.notna(i) else np.nan)

    return out
