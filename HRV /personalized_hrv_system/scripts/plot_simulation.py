#!/usr/bin/env python3
"""Generate and plot the simulation scenarios (A-F) to PNG files.

Usage:
    python scripts/plot_simulation.py
    python scripts/plot_simulation.py --scenario B_hr_spike --duration-h 2
    python scripts/plot_simulation.py --out-dir plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.score_scenario import score_scenario  # noqa: E402
from src.simulation.simulator import SCENARIOS, generate_scenario  # noqa: E402

HORIZONS = [60, 300, 600]


def plot_scenario(name: str, cfg: dict, duration_h: float, out_dir: Path):
    df0 = generate_scenario(name, duration_h=duration_h, fs_hz=cfg["simulation"]["sample_rate_hz"])
    res = score_scenario(df0, cfg, HORIZONS)
    df, valid = res["df"], res["valid"]
    t = res["t"]
    labels = res["labels"].astype(bool)
    combined, alert = res["score"], res["alert"]
    sc = res["scorecard"]

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(t, res["hr"], label="Actual HR", color="tab:red")
    axes[0].plot(t, res["y_pred_1min"], label="Persistence forecast (1min)", color="tab:blue", alpha=0.6)
    axes[0].fill_between(t, *axes[0].get_ylim(), where=labels, color="orange", alpha=0.15, label="Ground-truth anomaly")
    axes[0].set_ylabel("HR (bpm)"); axes[0].legend(loc="upper right"); axes[0].set_title(f"Scenario: {name}")

    axes[1].plot(t, df["ACC_mag"].to_numpy()[valid], color="tab:green"); axes[1].set_ylabel("ACC mag (g)")
    axes[2].plot(t, df["TEMP"].to_numpy()[valid], color="tab:purple"); axes[2].set_ylabel("TEMP (degC)")
    axes[3].plot(t, df["RMSSD"].to_numpy()[valid], color="tab:brown"); axes[3].set_ylabel("RMSSD (ms)")

    axes[4].plot(t, combined, label="Anomaly score", color="black")
    axes[4].axhline(res["threshold"], color="red", linestyle="--", label="Alert threshold (calibrated)")
    top = combined.max() * 1.05 if combined.max() > 0 else 1
    axes[4].fill_between(t, 0, top, where=alert, color="red", alpha=0.2, label="Alert raised")
    axes[4].set_ylabel("Anomaly score"); axes[4].set_xlabel("Time"); axes[4].legend(loc="upper right")

    fig.suptitle(f"Scenario {name} — ROC-AUC={sc['roc_auc']:.3f}  PR-AUC={sc['pr_auc']:.3f}  "
                 f"alert rate={alert.mean()*100:.2f}%  FA/hr={sc['false_alerts_per_hour']:.2f}", y=1.02)

    fig.tight_layout()
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None, choices=list(SCENARIOS), help="Plot only this scenario")
    parser.add_argument("--duration-h", type=float, default=None)
    parser.add_argument("--out-dir", default=str(ROOT / "plots"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    duration_h = args.duration_h or cfg["simulation"]["duration_h"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [args.scenario] if args.scenario else list(SCENARIOS)
    for name in names:
        path = plot_scenario(name, cfg, duration_h, out_dir)
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
