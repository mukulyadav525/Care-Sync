#!/usr/bin/env python3
"""Feature-group ablation (Tier-2 #13): is each modality actually helping?

Trains the pooled multi-dataset model repeatedly, each time DROPPING one feature
group (BVP / EDA / TEMP / ACC / HRV / subject-baseline), and compares held-out
test MAE/RMSE against the full-feature model. If a group doesn't improve (or
hurts) held-out performance, drop it.

Usage:
    python scripts/run_ablation.py --model xgboost
    python scripts/run_ablation.py --model tcn --groups bvp eda temp
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import multi_dataset_pipeline as mdp  # noqa: E402

# feature-name prefixes that identify each modality group
GROUP_PREFIXES = {
    "bvp": ("BVP",),
    "eda": ("EDA",),
    "temp": ("TEMP",),
    "acc": ("ACC", "activity"),
    "hrv": ("RMSSD", "SDNN", "pNN50", "HR_from_IBI"),
    "baseline": ("HR_ctx", "HR_minus_baseline", "HR_z_rolling", "RMSSD_minus", "RMSSD_z", "RMSSD_ctx"),
}


def _patch_drop(group_prefixes):
    """Monkey-patch numeric_feature_columns to drop a feature group."""
    from src.features import build_features
    orig = build_features.numeric_feature_columns

    def patched(table, cfg):
        cols = orig(table, cfg)
        return [c for c in cols if not any(c.startswith(p) for p in group_prefixes)]
    build_features.numeric_feature_columns = patched
    mdp.build_features.numeric_feature_columns = patched
    return orig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["tcn", "lstm", "gru", "transformer", "xgboost"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "config_multi.yaml"))
    parser.add_argument("--groups", nargs="*", default=list(GROUP_PREFIXES), help="groups to ablate")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = ROOT / "models" / "ablation"

    from src.features import build_features
    rows = []

    # baseline: full feature set
    res = mdp.run_multi_dataset(copy.deepcopy(cfg), ROOT, out / "full", model_type=args.model)
    full = res["test"]["forecast_report"]["overall"]
    rows.append(("full", full["MAE"], full["RMSE"], full["R2"]))

    for g in args.groups:
        prefixes = GROUP_PREFIXES[g]
        orig = _patch_drop(prefixes)
        try:
            res = mdp.run_multi_dataset(copy.deepcopy(cfg), ROOT, out / f"no_{g}", model_type=args.model)
            m = res["test"]["forecast_report"]["overall"]
            rows.append((f"-{g}", m["MAE"], m["RMSE"], m["R2"]))
        finally:
            build_features.numeric_feature_columns = orig
            mdp.build_features.numeric_feature_columns = orig

    print(f"\n===== Feature ablation ({args.model}) — vs full =====")
    print(f"{'variant':<12} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'dMAE':>8}")
    base_mae = rows[0][1]
    for name, mae, rmse, r2 in rows:
        d = mae - base_mae
        flag = "  (worse w/o)" if name != "full" and d < 0 else ""
        print(f"{name:<12} {mae:>8.3f} {rmse:>8.3f} {r2:>8.3f} {d:>+8.3f}{flag}")
    print("\nInterpretation: dMAE < 0 means dropping the group LOWERED error -> group may be noise.")


if __name__ == "__main__":
    main()
