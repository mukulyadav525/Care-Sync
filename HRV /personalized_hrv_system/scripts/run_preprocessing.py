#!/usr/bin/env python3
"""Build and save the feature table for one subject.

Usage:
    python scripts/run_preprocessing.py --subject S01 --out processed/S01_features.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.train_pipeline import load_and_featurize  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject folder name, e.g. S01")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_root = (ROOT.parent / cfg["data"]["raw_root"]).resolve()
    subject_dir = raw_root / args.subject

    table, _ = load_and_featurize(subject_dir, cfg)
    print(f"Feature table for {args.subject}: {table.shape[0]} rows x {table.shape[1]} cols")
    print(f"Valid (non-NaN) rows: {table.dropna().shape[0]}")

    out_path = Path(args.out) if args.out else ROOT / "processed" / f"{args.subject}_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
