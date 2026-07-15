#!/usr/bin/env python3
"""Train ONE global HR/HRV forecaster on a subject-level 3-way split (train/val/
test) pooled across Dataset3 (Stress-Predict) + WESAD, then report held-out TEST
metrics: per-horizon HR & RMSSD, prediction-interval coverage, per-dataset and
per-subject breakdowns.

Usage:
    python scripts/run_multi_dataset.py --model tcn
    python scripts/run_multi_dataset.py --model xgboost
    python scripts/run_multi_dataset.py --model tcn --repeats 5   # leave-subjects-out CV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import metrics as eval_metrics  # noqa: E402
from src.pipeline.multi_dataset_pipeline import (  # noqa: E402
    run_leave_one_dataset_out, run_multi_dataset, run_repeated_splits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tcn", choices=["tcn", "lstm", "gru", "transformer", "xgboost"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "config_multi.yaml"))
    parser.add_argument("--out", default=None, help="Output dir (default: models/multi/<model>)")
    parser.add_argument("--repeats", type=int, default=1, help="N>1 = repeated random subject splits (CV)")
    parser.add_argument("--lodo", action="store_true", help="leave-one-dataset-out cross-dataset eval")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out) if args.out else ROOT / "models" / "multi" / args.model

    if args.lodo:
        res = run_leave_one_dataset_out(cfg, ROOT, out_dir, model_type=args.model)
        print(f"\n===== Leave-one-dataset-out — {args.model} =====")
        for ds, m in res.items():
            print(f"  holdout={ds:<16} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}  r={m['pearson_r']:.3f}")
        return

    if args.repeats > 1:
        cv = run_repeated_splits(cfg, ROOT, out_dir, model_type=args.model, n_repeats=args.repeats)
        print(f"\n===== Leave-subjects-out CV ({cv['n_repeats']} random splits) — {args.model} =====")
        for k, v in cv["aggregate"].items():
            print(f"  {k:>9}: {v['mean']:.3f} +/- {v['std']:.3f}")
        return

    result = run_multi_dataset(cfg, ROOT, out_dir, model_type=args.model)
    s = result["summary"]

    print("\n================ POOLED MULTI-DATASET (subject-level 3-way split) ================")
    print(f"Datasets        : {', '.join(s['datasets'])}")
    print(f"Split           : train={s['n_train_subjects']} | val={s['n_val_subjects']} | test={s['n_test_subjects']} subjects (seed {s['seed']})")
    print(f"Train windows   : {s['n_train_windows']}   Features: {len(s['feature_cols'])}   Model: {args.model}")
    print(f"Targets         : {s['target_cols']}")
    print("\nTRAIN:", ", ".join(f"{d}/{i}" for d, i in result["train_subjects"]))
    print("VAL  :", ", ".join(f"{d}/{i}" for d, i in result["val_subjects"]))
    print("TEST :", ", ".join(f"{d}/{i}" for d, i in result["test_subjects"]))

    te = result["test"]
    print("\n" + eval_metrics.format_forecast_report(te["forecast_report"],
          title=f"HELD-OUT TEST (per horizon, HR & RMSSD) — {args.model}"))
    print("\n" + eval_metrics.format_per_subject(te["per_subject"],
          title="TEST per-subject metric distribution"))
    for ds, rep in te["per_dataset"].items():
        print("\n" + eval_metrics.format_forecast_report(rep, title=f"per-dataset: {ds}"))

    print(f"\nSaved global model + split.json -> {out_dir}")


if __name__ == "__main__":
    main()
