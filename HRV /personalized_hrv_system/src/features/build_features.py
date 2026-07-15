"""Orchestrates feature extraction across all signal modules into one feature table."""
from __future__ import annotations

import pandas as pd

from . import (acc_features, activity_gating, bvp_features, circadian, eda_features, hr_features, ibi_features,
               recovery_features, state_classifier, subject_baseline, temp_features, time_features)


def build_feature_table(synced: dict, cfg: dict) -> pd.DataFrame:
    """Build the full per-second feature table for one subject.

    Parameters
    ----------
    synced : dict returned by `src.data.sync.build_synced_frame`
    cfg : loaded YAML config

    Returns
    -------
    DataFrame indexed by the 1Hz grid, containing all engineered features plus
    forecast targets `HR_target_{h}s` for each horizon in cfg.model.horizons_s.
    """
    grid = synced["grid"]
    index = grid.index

    # HR + IBI + time are always available; the rest depend on which sensors a
    # given dataset provides, so include them only when their input
    # columns/signals are present.
    pieces = [
        grid,
        hr_features.compute_hr_features(grid, cfg),
        ibi_features.compute_ibi_features(index, synced["ibi_clean"], cfg),
        time_features.compute_time_features(index),
    ]
    if "ACC_mag_mean" in grid.columns:
        pieces.append(acc_features.compute_acc_features(grid, cfg))
    if len(synced.get("bvp_clean", [])) > 0:
        pieces.append(bvp_features.compute_bvp_features(index, synced["bvp_clean"], cfg))
    if "TEMP" in grid.columns and grid["TEMP"].notna().any():
        pieces.append(temp_features.compute_temp_features(grid, cfg))
    if "EDA" in grid.columns and grid["EDA"].notna().any():
        pieces.append(eda_features.compute_eda_features(grid, cfg))
    table = pd.concat(pieces, axis=1)

    # circadian baseline depends on HR, computed after the base table exists
    table = pd.concat([table, circadian.compute_circadian_baseline(table)], axis=1)

    # physiological state classifier depends on activity + HR + circadian features
    table = pd.concat([table, state_classifier.classify_physiological_state(table)], axis=1)

    # heart-rate-recovery features depend on the physiological state
    table = pd.concat([table, recovery_features.compute_recovery_features(table)], axis=1)

    # Tier-2: subject-relative baselines + long-context summaries, and
    # activity-conditioned HR deviation features.
    table = pd.concat([table, subject_baseline.compute_subject_baseline_features(table, cfg)], axis=1)
    table = pd.concat([table, activity_gating.compute_activity_gating_features(table)], axis=1)

    # forecast targets: future HR (and a short-window RMSSD, for HRV-aware
    # forecasting) at each horizon. `RMSSD_valid_fraction_*` must never be
    # mistaken for the RMSSD signal.
    rmssd_col = next((c for c in table.columns
                      if c.startswith("RMSSD_") and "valid_fraction" not in c), None)
    mcfg = cfg["model"]
    for h in mcfg["horizons_s"]:
        table[f"HR_target_{h}s"] = table["HR"].shift(-h)
        if rmssd_col is not None:
            table[f"RMSSD_target_{h}s"] = table[rmssd_col].shift(-h)
        # Tier-2 #11: delta-from-now targets teach the model to predict CHANGE
        if mcfg.get("predict_deltas", False):
            table[f"HR_delta_target_{h}s"] = table["HR"].shift(-h) - table["HR"]
            if rmssd_col is not None:
                table[f"RMSSD_delta_target_{h}s"] = table[rmssd_col].shift(-h) - table[rmssd_col]
        # Tier-3 #17: also forecast other vitals (multi-task shared representation)
        if mcfg.get("predict_vitals", False):
            for vit in ("TEMP", "EDA"):
                if vit in table.columns:
                    table[f"{vit}_target_{h}s"] = table[vit].shift(-h)

    return table


def build_target_cols(table: pd.DataFrame, cfg: dict) -> list[str]:
    """Assemble the forecast target columns from config flags, in a stable order:
    HR (always) -> RMSSD -> HR/RMSSD deltas -> other vitals."""
    horizons = cfg["model"]["horizons_s"]
    mcfg = cfg["model"]
    cols = [f"HR_target_{h}s" for h in horizons]
    if mcfg.get("predict_rmssd", False):
        cols += [f"RMSSD_target_{h}s" for h in horizons]
    if mcfg.get("predict_deltas", False):
        cols += [f"HR_delta_target_{h}s" for h in horizons]
        if mcfg.get("predict_rmssd", False):
            cols += [f"RMSSD_delta_target_{h}s" for h in horizons]
    if mcfg.get("predict_vitals", False):
        for vit in ("TEMP", "EDA"):
            cols += [f"{vit}_target_{h}s" for h in horizons]
    return [c for c in cols if c in table.columns]


def numeric_feature_columns(table: pd.DataFrame, cfg: dict) -> list[str]:
    """Columns to feed to the model (excludes targets and categorical/name columns)."""
    exclude_prefixes = ("HR_target_", "RMSSD_target_", "HR_delta_target_",
                        "RMSSD_delta_target_", "TEMP_target_", "EDA_target_")
    exclude_exact = {"activity_bucket_name", "physio_state_name"}
    cols = []
    for c in table.columns:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(table[c]):
            continue
        cols.append(c)
    return cols
