#!/usr/bin/env python3
"""Self-supervised masked-reconstruction pretraining of a TCN backbone (Tier-3 #18).

Pretrain on unlabelled windows pooled across subjects, then warm-start a personal
forecaster with:  run_training.py --subject S01 --model tcn --ssl-init models/ssl

Usage:
    python scripts/run_ssl_pretraining.py --all
    python scripts/run_ssl_pretraining.py --subjects S01,S02,S03
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.ssl_pretrain_pipeline import run_ssl_pretraining  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default=None, help="Comma-separated subject ids")
    parser.add_argument("--all", action="store_true", help="Use every S* folder under raw_root")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--out", default=None, help="Output dir (default: models/ssl)")
    parser.add_argument("--mask-prob", type=float, default=0.25)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_root = (ROOT.parent / cfg["data"]["raw_root"]).resolve()
    if args.all:
        subject_dirs = sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("S"))
    elif args.subjects:
        subject_dirs = [raw_root / s.strip() for s in args.subjects.split(",")]
    else:
        parser.error("Provide --subjects S01,... or --all")

    out_dir = Path(args.out) if args.out else ROOT / "models" / "ssl"
    res = run_ssl_pretraining(subject_dirs, cfg, out_dir, mask_prob=args.mask_prob)
    print(f"\nSSL pretraining done: {res['n_windows']} windows, recon_mse={res['recon_mse']:.4f}")
    print(f"Saved encoder -> {out_dir}/ssl_encoder.pt")
    print(f"Warm-start a model with:  scripts/run_training.py --model tcn --ssl-init {out_dir}")


if __name__ == "__main__":
    main()
