#!/usr/bin/env python3
"""Train a personalized HR forecaster for one subject.

Usage:
    python scripts/run_training.py --subject S01 --model tcn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import metrics as eval_metrics  # noqa: E402
from src.pipeline.train_pipeline import run_training  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject folder name, e.g. S01")
    parser.add_argument("--model", default="tcn", choices=["tcn", "lstm", "gru", "transformer", "xgboost"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--out", default=None, help="Output directory for model artifacts")
    parser.add_argument("--ssl-init", default=None, help="SSL-pretrained encoder dir to warm-start a TCN (Tier-3 #18)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_root = (ROOT.parent / cfg["data"]["raw_root"]).resolve()
    subject_dir = raw_root / args.subject

    out_dir = Path(args.out) if args.out else ROOT / "models" / args.subject / args.model
    result = run_training(subject_dir, cfg, out_dir, model_type=args.model, ssl_init_dir=args.ssl_init)

    print(f"\nSaved model artifacts -> {out_dir}")
    if result.get("regression_report") is not None:
        print(eval_metrics.format_regression(
            result["regression_report"],
            title=f"HR forecast — {args.subject} / {args.model}",
        ))
    else:
        print("Test-set forecast metrics (per horizon):")
        for k, v in result["test_metrics"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
