"""Physiological state classifier (DESIGN.md addendum: state model).

Assigns each 1Hz sample one of: sleep, rest, focused_work, walking, exercise,
recovery, unknown. This gives the anomaly layer context: a HR spike during
`exercise` is expected, the same spike during `sleep` is not.

All inputs are causal (rolling windows / shifts only), so this can run in
real time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATE_LABELS = ["sleep", "rest", "focused_work", "walking", "exercise", "recovery", "unknown"]
SLEEP, REST, FOCUSED_WORK, WALKING, EXERCISE, RECOVERY, UNKNOWN = range(7)

# activity_bucket values from acc_features.ACTIVITY_LABELS
_ACC_REST, _ACC_LIGHT, _ACC_MODERATE, _ACC_VIGOROUS = 0, 1, 2, 3

NIGHT_HOURS = set(range(23, 24)) | set(range(0, 6))  # 23:00 - 05:59
RECOVERY_LOOKBACK_S = 600  # how long after exercise we consider "recovery"


def classify_physiological_state(table: pd.DataFrame) -> pd.DataFrame:
    """Requires columns: activity_bucket, HR, HR_roll_mean_60s, HR_trend_300s,
    HR_circadian_mean, EDA_phasic_activity_60s. Missing columns degrade gracefully
    to `unknown`.

    Returns a DataFrame with `state` (numeric code) and `state_name` (string,
    excluded from model features by `numeric_feature_columns`).
    """
    n = len(table)
    required = ["activity_bucket", "HR", "HR_roll_mean_60s", "HR_trend_300s", "HR_circadian_mean"]
    if not all(c in table.columns for c in required):
        state = np.full(n, UNKNOWN)
        return _to_frame(table.index, state)

    activity = table["activity_bucket"].to_numpy()
    hr_now = table["HR_roll_mean_60s"].to_numpy()
    hr_trend = table["HR_trend_300s"].to_numpy()
    circadian_mean = table["HR_circadian_mean"].to_numpy()
    hour = table.index.hour.to_numpy()

    eda_arousal = table["EDA_phasic_activity_60s"] if "EDA_phasic_activity_60s" in table.columns else pd.Series(np.nan, index=table.index)
    eda_baseline = eda_arousal.rolling(3600, min_periods=300).median()
    eda_elevated = (eda_arousal > eda_baseline).fillna(False).to_numpy()

    was_exercising = (table["activity_bucket"] >= _ACC_MODERATE)
    recent_exercise = was_exercising.rolling(RECOVERY_LOOKBACK_S, min_periods=1).max().fillna(0).astype(bool).to_numpy()

    is_night = np.isin(hour, list(NIGHT_HOURS))
    above_baseline = hr_now > (circadian_mean + 3.0)

    state = np.full(n, UNKNOWN, dtype=int)
    valid = ~np.isnan(activity) & ~np.isnan(hr_now) & ~np.isnan(hr_trend) & ~np.isnan(circadian_mean)

    exercising = valid & (activity >= _ACC_MODERATE)
    state[exercising] = EXERCISE

    # HR still elevated above this person's circadian baseline shortly after an
    # exercise bout, regardless of momentary trend direction (HR can plateau
    # before it starts dropping).
    recovering = valid & ~exercising & recent_exercise & above_baseline
    state[recovering] = RECOVERY

    remaining = valid & (state == UNKNOWN)
    walking = remaining & (activity == _ACC_LIGHT)
    state[walking] = WALKING

    remaining = valid & (state == UNKNOWN)
    sleeping = remaining & (activity == _ACC_REST) & is_night & (hr_now <= circadian_mean)
    state[sleeping] = SLEEP

    remaining = valid & (state == UNKNOWN)
    focused = remaining & (activity == _ACC_REST) & ~is_night & eda_elevated
    state[focused] = FOCUSED_WORK

    remaining = valid & (state == UNKNOWN)
    state[remaining & (activity == _ACC_REST)] = REST

    return _to_frame(table.index, state)


def _to_frame(index: pd.Index, state: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["physio_state"] = state
    out["physio_state_name"] = [STATE_LABELS[s] for s in state]
    return out
