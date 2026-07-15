#!/usr/bin/env python3
"""Interactively vary "vitals" (activity, temperature, EDA/stress, spikes, drift,
noise) and see the resulting effect on HR.

Examples
--------
# Baseline normal day, 24h, default activity schedule
python scripts/explore_vitals.py

# Constant vigorous activity for 1 hour -> see HR ramp up and recover
python scripts/explore_vitals.py --duration-h 1 --activity-level 0.9

# Fever: +2 degC for 2 hours -> see resting HR rise ~5bpm
python scripts/explore_vitals.py --duration-h 2 --temp-offset 2.0 --activity-level 0.0

# High stress (EDA) at rest
python scripts/explore_vitals.py --duration-h 1 --activity-level 0.0 --eda-offset 1.5

# Sudden HR spike (e.g. panic/arrhythmia) at minute 10, for 2 hours total
python scripts/explore_vitals.py --duration-h 2 --hr-spike-amplitude 30 --hr-spike-time-h 0.167

# Gradual drift: +15 bpm over the trace
python scripts/explore_vitals.py --duration-h 3 --hr-drift-rate 5 --activity-level 0.1

# Combine everything
python scripts/explore_vitals.py --duration-h 2 --activity-level 0.6 --temp-offset 1.0 \
    --eda-offset 0.5 --noise-std 2.0 --out my_run.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.simulator import simulate_custom  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration-h", type=float, default=2.0, help="Trace duration in hours")
    p.add_argument("--baseline-hr", type=float, default=65.0, help="Resting circadian-baseline HR (bpm)")
    p.add_argument("--circadian-amplitude", type=float, default=8.0, help="Circadian HR swing amplitude (bpm)")
    p.add_argument("--activity-level", type=float, default=None,
                    help="Constant activity intensity in [0,1] (0=rest,1=vigorous). Omit for the default daily schedule.")
    p.add_argument("--activity-gain", type=float, default=70.0, help="bpm added at activity_level=1.0")
    p.add_argument("--temp-offset", type=float, default=0.0, help="Constant TEMP shift, degC (e.g. +2 = fever)")
    p.add_argument("--temp-gain", type=float, default=2.5, help="bpm per degC of TEMP above baseline")
    p.add_argument("--eda-offset", type=float, default=0.0, help="Constant EDA shift (arousal/stress)")
    p.add_argument("--eda-gain", type=float, default=10.0, help="bpm per unit of EDA above baseline")
    p.add_argument("--hr-spike-amplitude", type=float, default=0.0, help="Add a Gaussian HR pulse, bpm")
    p.add_argument("--hr-spike-time-h", type=float, default=None, help="Hour (from start) where the spike is centered (default: midpoint)")
    p.add_argument("--hr-spike-width-s", type=float, default=30.0, help="Spike width (seconds)")
    p.add_argument("--hr-drift-rate", type=float, default=0.0, help="Linear HR drift, bpm per hour")
    p.add_argument("--noise-std", type=float, default=1.5, help="HR measurement noise std (bpm)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "plots" / "explore_vitals.png"))
    args = p.parse_args()

    df = simulate_custom(
        duration_h=args.duration_h,
        baseline_hr=args.baseline_hr,
        circadian_amplitude=args.circadian_amplitude,
        activity_level=args.activity_level,
        activity_gain_bpm=args.activity_gain,
        temp_offset=args.temp_offset,
        temp_gain_bpm_per_degC=args.temp_gain,
        eda_offset=args.eda_offset,
        eda_gain_bpm=args.eda_gain,
        hr_spike_amplitude=args.hr_spike_amplitude,
        hr_spike_time_h=args.hr_spike_time_h,
        hr_spike_width_s=args.hr_spike_width_s,
        hr_drift_rate_bpm_per_h=args.hr_drift_rate,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    print(df.describe())

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(df.index, df["HR"], label="HR (noisy)", color="tab:red")
    axes[0].plot(df.index, df["HR_target"], label="HR target (no noise)", color="tab:red", alpha=0.4, linestyle="--")
    axes[0].set_ylabel("HR (bpm)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Vitals -> HR exploration")

    axes[1].plot(df.index, df["activity_intensity"], color="tab:green")
    axes[1].set_ylabel("Activity\n(0-1)")

    axes[2].plot(df.index, df["TEMP"], color="tab:purple")
    axes[2].set_ylabel("TEMP\n(degC)")

    axes[3].plot(df.index, df["EDA"], color="tab:orange")
    axes[3].set_ylabel("EDA")

    axes[4].plot(df.index, df["RMSSD"], color="tab:brown")
    axes[4].set_ylabel("RMSSD\n(ms)")
    axes[4].set_xlabel("Time")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
