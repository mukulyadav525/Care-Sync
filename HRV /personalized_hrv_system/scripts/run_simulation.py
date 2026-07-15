#!/usr/bin/env python3
"""Evaluate the anomaly detector across simulated scenarios (A-G).

Uses the Tier-1/2 detector (uncertainty-normalized residuals + SQI down-weight +
threshold calibrated on each scenario's NORMAL windows + hysteresis) via the
shared `score_scenario` helper, and prints a per-scenario scorecard:
ROC-AUC, PR-AUC, event recall, time-to-detect, false-alerts/hour, normal alert rate.

Usage:
    python scripts/run_simulation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.score_scenario import score_scenario  # noqa: E402
from src.simulation.simulator import SCENARIOS, generate_scenario  # noqa: E402

HORIZONS = [60, 300, 600]


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))

    print(f"{'Scenario':<16} {'ROC':>5} {'PR':>5} {'evRec':>6} {'TTD':>6} {'FA/hr':>7} {'normFA':>7} {'thr':>5}")
    for name in SCENARIOS:
        df = generate_scenario(name, duration_h=cfg["simulation"]["duration_h"],
                               fs_hz=cfg["simulation"]["sample_rate_hz"])
        res = score_scenario(df, cfg, HORIZONS)
        sc = res["scorecard"]
        print(f"{name:<16} {sc['roc_auc']:>5.3f} {sc['pr_auc']:>5.3f} {sc['event_recall']:>6.2f} "
              f"{sc['median_time_to_detect_s']:>6.0f} {sc['false_alerts_per_hour']:>7.2f} "
              f"{sc['normal_alert_rate']:>7.3f} {res['threshold']:>5.1f}")


if __name__ == "__main__":
    main()
