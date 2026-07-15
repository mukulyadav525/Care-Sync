#!/usr/bin/env python3
"""Pretrain a global (population) forecaster across multiple subjects.

Usage:
    python scripts/run_pretraining.py --subjects S01,S02,S03 --model tcn
    python scripts/run_pretraining.py --all --model tcn   # use every SXX folder found
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.pretrain_pipeline import run_pretraining  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default=None, help="Comma-separated subject folder names, e.g. S01,S02,S03")
    parser.add_argument("--all", action="store_true", help="Use every SXX folder found under the raw data root")
    parser.add_argument("--model", default="tcn", choices=["tcn", "lstm", "gru", "transformer"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--out", default=None, help="Output directory for the global model artifacts")
    parser.add_argument("--raw-root", default=None,
                         help="Override cfg data.raw_root - e.g. point at Care-Sync real session storage "
                              "(backend/Users) instead of the demo dataset.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_root = Path(args.raw_root).resolve() if args.raw_root else (ROOT.parent / cfg["data"]["raw_root"]).resolve()

    if args.all:
        subject_dirs = sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("S"))
    elif args.subjects:
        subject_dirs = [raw_root / s.strip() for s in args.subjects.split(",")]
    else:
        parser.error("Provide --subjects S01,S02,... or --all")

    out_dir = Path(args.out) if args.out else ROOT / "models" / "global" / args.model
    result = run_pretraining(subject_dirs, cfg, out_dir, model_type=args.model)

    print(f"\nPretrained on {result['n_subjects']} subjects -> {out_dir}")
    print(f"Feature columns ({len(result['feature_cols'])}): {result['feature_cols']}")


if __name__ == "__main__":
    main()
