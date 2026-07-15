#!/usr/bin/env python3
"""Run inference + anomaly scoring for a trained personal model over a subject's data.

Usage:
    python scripts/run_inference.py --subject S01 --model tcn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.inference_pipeline import load_model, run_inference  # noqa: E402
from src.pipeline.train_pipeline import load_and_featurize  # noqa: E402
from src.personalization.digital_twin import load_digital_twin  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--model", default="tcn", choices=["tcn", "lstm", "gru", "transformer"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_root = (ROOT.parent / cfg["data"]["raw_root"]).resolve()
    subject_dir = raw_root / args.subject

    model_dir = Path(args.model_dir) if args.model_dir else ROOT / "models" / args.subject / args.model
    model, scaler, meta = load_model(model_dir)

    twin_path = ROOT / "models" / args.subject / "digital_twin.json"
    twin = load_digital_twin(twin_path) if twin_path.exists() else None
    if twin is not None:
        print(f"Loaded digital twin -> {twin_path}")

    table, _ = load_and_featurize(subject_dir, cfg)
    results = run_inference(table, model, scaler, meta, cfg, twin=twin)
    if results.empty:
        print("No scored windows (recording too short for the input window)."); return

    from src.anomaly.detector import alert_stats  # noqa: PLC0415
    n_alerts = int(results["alert"].sum())
    print(f"Scored {len(results)} timesteps, {n_alerts} alert timesteps "
          f"({100*n_alerts/max(1,len(results)):.2f}%)  threshold={results['alert_threshold'].iloc[0]:.2f}")
    st = alert_stats(results["alert"].to_numpy(), fs=1.0)
    print(f"  alert events: {st['n_events']}  alerts/hour: {st['alerts_per_hour']:.2f}  "
          f"mean duration: {st['mean_alert_duration_s']:.0f}s")

    out_path = Path(args.out) if args.out else ROOT / "processed" / f"{args.subject}_{args.model}_inference.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
