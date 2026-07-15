#!/usr/bin/env python3
"""Incrementally update a personal model with a subject's most recent data.

Intended to be run periodically (e.g. via cron) so the model tracks slow
changes in someone's baseline fitness/physiology over time.

Usage:
    python scripts/run_online_update.py --subject S01 --model-dir models/S01/tcn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.online_update_pipeline import run_online_update  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject folder name, e.g. S01")
    parser.add_argument("--model-dir", required=True, help="Directory with the existing model artifacts")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--out", default=None, help="Output directory (default: overwrite --model-dir)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_root = (ROOT.parent / cfg["data"]["raw_root"]).resolve()
    subject_dir = raw_root / args.subject

    result = run_online_update(subject_dir, cfg, args.model_dir, out_dir=args.out)

    print(f"\nOnline update done using {result['n_recent_windows']} recent windows.")
    print("Validation metrics on the held-out tail of the recent window:")
    for k, v in result["val_metrics"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
