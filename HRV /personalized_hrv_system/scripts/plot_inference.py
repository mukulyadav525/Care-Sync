#!/usr/bin/env python3
"""Plot trained-model forecasts vs actual HR and the anomaly score timeline.

Usage:
    python scripts/run_training.py --subject S01 --model tcn   # train first
    python scripts/run_inference.py --subject S01 --model tcn  # generates the CSV
    python scripts/plot_inference.py --subject S01 --model tcn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--model", default="tcn")
    parser.add_argument("--csv", default=None, help="Path to inference CSV (default: processed/<subject>_<model>_inference.csv)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--horizon", type=int, default=60, help="Which horizon (s) to plot in the top panel")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else ROOT / "processed" / f"{args.subject}_{args.model}_inference.csv"
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    h = args.horizon
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df.index, df[f"y_true_{h}s"], label=f"Actual HR (+{h}s)", color="tab:red")
    axes[0].plot(df.index, df[f"y_pred_{h}s"], label=f"Predicted HR (+{h}s)", color="tab:blue")
    upper = df[f"y_pred_{h}s"] + 1.96 * df[f"y_std_{h}s"]
    lower = df[f"y_pred_{h}s"] - 1.96 * df[f"y_std_{h}s"]
    axes[0].fill_between(df.index, lower, upper, color="tab:blue", alpha=0.2, label="95% prediction interval")
    axes[0].set_ylabel("HR (bpm)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{args.subject} ({args.model}) — forecast horizon = {h}s")

    axes[1].plot(df.index, df[f"residual_{h}s"], color="tab:gray", label=f"Residual (+{h}s)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Residual (bpm)")
    axes[1].legend(loc="upper right")

    axes[2].plot(df.index, df["anomaly_score"], color="black", label="Combined anomaly score")
    alert_mask = df["alert"].astype(bool)
    axes[2].fill_between(df.index, 0, df["anomaly_score"].max() * 1.05, where=alert_mask,
                          color="red", alpha=0.2, label="Alert")
    axes[2].set_ylabel("Anomaly score")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    out_path = Path(args.out) if args.out else ROOT / "plots" / f"{args.subject}_{args.model}_inference.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
