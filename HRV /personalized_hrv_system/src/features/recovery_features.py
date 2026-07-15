"""Heart-rate-recovery (HRR) features (DESIGN.md addendum: recovery & fitness tracking).

Tracks the peak HR of the most recent exercise bout and how far/fast HR has
dropped from that peak since. A faster drop (higher `hr_recovery_rate_bpm_per_min`)
indicates better cardiovascular fitness; persistently slow recovery over weeks
can flag declining fitness or incomplete recovery.

All outputs default to 0 outside of/before any exercise bout, so this never
introduces NaNs that would cause `make_windows` to drop samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .state_classifier import EXERCISE, RECOVERY


def compute_recovery_features(table: pd.DataFrame) -> pd.DataFrame:
    """Requires columns: physio_state, HR_roll_mean_60s (falls back to HR)."""
    n = len(table)
    if "physio_state" not in table.columns:
        zeros = np.zeros(n)
        out = pd.DataFrame(index=table.index)
        out["exercise_peak_hr"] = zeros
        out["hr_recovery_bpm"] = zeros
        out["hr_recovery_rate_bpm_per_min"] = zeros
        out["time_since_exercise_peak_s"] = zeros
        return out

    state = table["physio_state"].to_numpy()
    hr = (table["HR_roll_mean_60s"] if "HR_roll_mean_60s" in table.columns else table["HR"]).to_numpy()

    peak = np.zeros(n)
    time_since_peak = np.zeros(n)

    current_peak = 0.0
    counter = 0
    in_bout = False
    for i in range(n):
        s = state[i]
        h = hr[i]
        if np.isnan(h):
            h = current_peak
        if s == EXERCISE:
            current_peak = max(current_peak, h) if in_bout else h
            in_bout = True
            counter = 0
        elif s == RECOVERY and in_bout:
            counter += 1
        else:
            in_bout = False
            current_peak = h
            counter = 0
        peak[i] = current_peak
        time_since_peak[i] = counter

    # NaN-safe: where HR was missing, treat current value as the peak (0 recovery)
    hr_safe = np.where(np.isnan(hr), peak, hr)
    hr_recovery_bpm = np.maximum(0.0, peak - hr_safe)
    minutes = np.maximum(time_since_peak, 1.0) / 60.0
    hr_recovery_rate = np.where(time_since_peak > 0, hr_recovery_bpm / minutes, 0.0)

    out = pd.DataFrame(index=table.index)
    out["exercise_peak_hr"] = peak
    out["hr_recovery_bpm"] = hr_recovery_bpm
    out["hr_recovery_rate_bpm_per_min"] = hr_recovery_rate
    out["time_since_exercise_peak_s"] = time_since_peak
    return out
