#!/usr/bin/env python3
"""Per-subject error analysis (Tier-3 #20).

For a trained single-subject model, breaks the held-out forecast error down by
the dimensions that actually explain failures: activity level, time-of-day,
signal quality (SQI), and prediction horizon. Writes a CSV + a multi-panel PNG.

Usage:
    python scripts/run_error_analysis.py --subject S01 --model tcn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly import signal_quality  # noqa: E402
from src.models import datasets  # noqa: E402
from src.pipeline.inference_pipeline import load_model  # noqa: E402
from src.pipeline.train_pipeline import load_and_featurize  # noqa: E402


def _bucketed(df, by, err_col):
    g = df.groupby(by)[err_col]
    return g.agg(["mean", "std", "count"]).rename(columns={"mean": "MAE", "std": "err_std", "count": "n"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--model", default="tcn")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_root = (ROOT.parent / cfg["data"]["raw_root"]).resolve()
    model_dir = Path(args.model_dir) if args.model_dir else ROOT / "models" / args.subject / args.model
    model, scaler, meta = load_model(model_dir)

    table, _ = load_and_featurize(raw_root / args.subject, cfg)
    sqi = signal_quality.compute_signal_quality(table)
    table = table.join(sqi[["sqi_overall"]])

    feature_cols, target_cols, seq_len = meta["feature_cols"], meta["target_cols"], meta["seq_len"]
    X, y, end_idx = datasets.make_windows(table, feature_cols, target_cols, seq_len, stride=1)
    if len(X) == 0:
        print("No windows for this subject."); return
    Xs = datasets.apply_scaler(X, scaler)

    from src.models import train as train_mod
    if meta["model_type"] == "xgboost":
        mean, _ = model.predict(Xs[:, 0, :])
    else:
        mean_s, _ = train_mod.predict(model, Xs)
        ts = meta.get("_target_scaler")
        mean = datasets.inverse_target_scaler(mean_s, ts) if ts is not None else mean_s

    # focus on the 1-min HR horizon for the breakdown
    hr_idx = next((i for i, c in enumerate(target_cols) if c.startswith("HR_target_")), 0)
    abs_err = np.abs(y[:, hr_idx] - mean[:, hr_idx])

    df = pd.DataFrame({"abs_err": abs_err}, index=table.index[end_idx])
    df["hour"] = df.index.hour
    df["activity"] = table["physio_state_name"].to_numpy()[end_idx] if "physio_state_name" in table.columns else "unknown"
    df["sqi_bin"] = pd.cut(table["sqi_overall"].to_numpy()[end_idx], [0, 0.5, 0.8, 1.01],
                           labels=["artifact", "noisy", "good"])

    print(f"\n===== Error analysis — {args.subject}/{args.model} (HR +{target_cols[hr_idx]}) =====")
    print(f"Overall MAE: {abs_err.mean():.3f} bpm  (n={len(abs_err)})")
    print("\nBy activity state:\n", _bucketed(df, "activity", "abs_err").round(2))
    print("\nBy signal quality:\n", _bucketed(df, "sqi_bin", "abs_err").round(2))
    print("\nBy hour of day (top-5 worst):\n",
          _bucketed(df, "hour", "abs_err").sort_values("MAE", ascending=False).head(5).round(2))

    out_csv = Path(args.out) if args.out else ROOT / "processed" / f"{args.subject}_{args.model}_error_analysis.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    print(f"\nSaved per-timestep error table -> {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        _bucketed(df, "activity", "abs_err")["MAE"].plot.bar(ax=axes[0], title="MAE by activity")
        _bucketed(df, "sqi_bin", "abs_err")["MAE"].plot.bar(ax=axes[1], title="MAE by signal quality", color="tab:orange")
        _bucketed(df, "hour", "abs_err")["MAE"].plot.line(ax=axes[2], marker="o", title="MAE by hour")
        for ax in axes:
            ax.set_ylabel("MAE (bpm)")
        fig.tight_layout()
        png = out_csv.with_suffix(".png")
        fig.savefig(png, dpi=110)
        print(f"Saved error-analysis plot -> {png}")
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
