"""Subject-relative baseline features (Tier-2 #9) and long-context summaries (#10).

Absolute HR/TEMP/RMSSD vary enormously between people, so a deviation from the
PERSON'S OWN recent baseline is usually more informative than the raw value.
This module adds, causally (backward-looking only):

  * <signal>_minus_baseline_<W>   : value - causal rolling mean over W seconds
  * <signal>_z_rolling_<W>        : (value - mean_W) / std_W   (personal z-score)

and multi-scale long-context summaries of HR / RMSSD over 15/30/60 min so the
model sees both the immediate trajectory and the slow baseline/recovery trend
without needing a 60-min raw input window:

  * HR_ctx_mean_<W> / HR_ctx_std_<W> / HR_ctx_slope_<W>

All deviation columns default to 0 (never NaN) so windowing never drops rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# signals to build subject-relative deviations for (only if present in the table)
_REL_SIGNALS = ["HR", "TEMP", "EDA"]
_DEFAULT_BASELINE_WINDOWS = [3600]          # 1 h personal baseline
_DEFAULT_LONGCTX_WINDOWS = [900, 1800, 3600]  # 15 / 30 / 60 min context


def compute_subject_baseline_features(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fcfg = cfg.get("features", {})
    base_windows = fcfg.get("baseline_windows_s", _DEFAULT_BASELINE_WINDOWS)
    ctx_windows = fcfg.get("longctx_windows_s", _DEFAULT_LONGCTX_WINDOWS)
    out = pd.DataFrame(index=table.index)

    # subject-relative deviations + personal z-scores
    rmssd_col = next((c for c in table.columns
                      if c.startswith("RMSSD_") and "valid_fraction" not in c), None)
    signals = [s for s in _REL_SIGNALS if s in table.columns]
    if rmssd_col is not None:
        signals.append(rmssd_col)

    for sig in signals:
        s = table[sig].astype(float)
        name = "RMSSD" if sig == rmssd_col else sig
        for w in base_windows:
            mean = s.rolling(w, min_periods=30).mean()
            std = s.rolling(w, min_periods=30).std()
            out[f"{name}_minus_baseline_{w}s"] = (s - mean).fillna(0.0)
            out[f"{name}_z_rolling_{w}s"] = ((s - mean) / std.replace(0, np.nan)).fillna(0.0)

    # multi-scale long-context summaries for HR and RMSSD (mean / std / slope).
    # All causal: min_periods=1 so values exist from the start WITHOUT backfilling
    # from the future, and the slope is a vectorised trailing endpoint slope
    # (value change per second over the window) — O(n), not a per-window Python loop.
    ctx_signals = [("HR", "HR")] + ([("RMSSD", rmssd_col)] if rmssd_col else [])
    for name, col in ctx_signals:
        s = table[col].astype(float)
        for w in ctx_windows:
            out[f"{name}_ctx_mean_{w}s"] = s.rolling(w, min_periods=1).mean().fillna(0.0)
            out[f"{name}_ctx_std_{w}s"] = s.rolling(w, min_periods=2).std().fillna(0.0)
            out[f"{name}_ctx_slope_{w}s"] = ((s - s.shift(w)) / float(w)).fillna(0.0)

    return out
