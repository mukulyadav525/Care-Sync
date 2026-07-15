"""Per-hour-of-day circadian baseline for HR (DESIGN.md addendum: circadian model).

For each hour-of-day bucket (0-23), maintains a causal (expanding, 1-step-shifted)
mean/std of HR. This gives:
  - HR_circadian_mean(t)  : "what is this person's normal HR at this time of day?"
  - HR_circadian_std(t)
  - HR_circadian_zscore(t): (HR(t) - HR_circadian_mean(t)) / HR_circadian_std(t)

With only a short single-session recording these are degenerate (a single hour
bucket), but the same code scales directly to 15+ days of continuous data, where
each hour bucket accumulates many prior observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_circadian_baseline(table: pd.DataFrame, min_periods: int = 10) -> pd.DataFrame:
    hr = table["HR"]
    hour = table.index.hour

    means = pd.Series(np.nan, index=table.index, dtype=float)
    stds = pd.Series(np.nan, index=table.index, dtype=float)

    for h in range(24):
        mask = hour == h
        if not mask.any():
            continue
        sub = hr[mask]
        means.loc[mask] = sub.expanding(min_periods=min_periods).mean().shift(1).to_numpy()
        stds.loc[mask] = sub.expanding(min_periods=min_periods).std().shift(1).to_numpy()

    out = pd.DataFrame(index=table.index)
    out["HR_circadian_mean"] = means
    out["HR_circadian_std"] = stds
    out["HR_circadian_zscore"] = (hr - means) / stds.clip(lower=1e-6)
    return out


def circadian_profile_table(table: pd.DataFrame, min_periods: int = 30) -> pd.DataFrame:
    """Full-history (non-causal) per-hour-of-day mean/std/count of HR — used to
    build the Personal Digital Twin's circadian profile (`digital_twin.py`)."""
    hr = table["HR"].dropna()
    hour = hr.index.hour
    grouped = hr.groupby(hour).agg(["mean", "std", "count"])
    grouped = grouped.reindex(range(24))
    grouped.index.name = "hour_of_day"
    grouped.columns = ["HR_mean", "HR_std", "n_samples"]
    grouped.loc[grouped["n_samples"] < min_periods, ["HR_mean", "HR_std"]] = np.nan
    return grouped
