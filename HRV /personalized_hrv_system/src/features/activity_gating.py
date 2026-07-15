"""Activity-aware gating features (Tier-2 #12).

High HR means very different things depending on motion: HR-high + ACC-high is
probably exercise; HR-high + ACC-low is more suspicious (stress / illness /
arousal). These features give the model (and the anomaly layer) an explicit
"how unusual is this HR FOR THIS activity level" signal.

Causal throughout: the expected-HR-per-activity-bucket baseline is an EXPANDING
mean keyed on the activity bucket (same trick as the circadian baseline), so it
only uses the past.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_activity_gating_features(table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=table.index)
    if "HR" not in table.columns:
        return out

    hr = table["HR"].astype(float).to_numpy()
    n = len(hr)

    # activity bucket: prefer the discrete bucket, else derive from ACC magnitude
    if "activity_bucket" in table.columns:
        bucket = table["activity_bucket"].fillna(0).astype(int).to_numpy()
    elif "ACC_mag_mean" in table.columns:
        amag = table["ACC_mag_mean"].fillna(0.0).to_numpy()
        bucket = np.digitize(amag, [0.05, 0.2, 0.5])  # rest/light/moderate/vigorous
    else:
        bucket = np.zeros(n, dtype=int)

    # causal expanding mean/var of HR per activity bucket (Welford per key).
    # NaN HR (sensor gaps) must NOT enter the running stats, otherwise the mean
    # becomes NaN and poisons every subsequent row.
    counts: dict[int, int] = {}
    means: dict[int, float] = {}
    m2: dict[int, float] = {}
    expected = np.zeros(n)
    resid = np.zeros(n)
    z = np.zeros(n)
    last_mean = 0.0
    for i in range(n):
        b = int(bucket[i])
        hr_i = hr[i]
        mean = means.get(b, hr_i if np.isfinite(hr_i) else last_mean)
        expected[i] = mean if np.isfinite(mean) else last_mean
        if not np.isfinite(hr_i):
            # gap: carry the expected value, no deviation, don't update stats
            resid[i] = 0.0
            z[i] = 0.0
            continue
        last_mean = mean
        resid[i] = hr_i - mean
        c = counts.get(b, 0)
        var = (m2.get(b, 0.0) / c) if c > 1 else 0.0
        std = np.sqrt(var)
        z[i] = resid[i] / std if std > 1e-6 else 0.0
        # update after using (strictly causal)
        c1 = c + 1
        delta = hr_i - mean
        mean1 = mean + delta / c1
        m2[b] = m2.get(b, 0.0) + delta * (hr_i - mean1)
        counts[b], means[b] = c1, mean1

    out["hr_expected_for_activity"] = expected
    out["hr_resid_for_activity"] = resid
    out["hr_z_for_activity"] = np.clip(z, -8, 8)

    # explicit "HR high while still" flag (low motion but elevated HR vs its activity baseline)
    low_motion = (bucket == 0).astype(float)
    out["hr_high_while_still"] = (low_motion * np.clip(z, 0, None)).astype(float)
    return out
