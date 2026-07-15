#!/usr/bin/env python3
"""Fine-tune a globally pretrained model on one subject's data.

Usage:
    python scripts/run_pretraining.py --all --model tcn
    python scripts/run_finetune.py --subject S01 --pretrained models/global/tcn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.finetune_pipeline import run_finetune  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject folder name, e.g. S01")
    parser.add_argument("--pretrained", required=True, help="Directory with the global model artifacts (from run_pretraining.py)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--out", default=None, help="Output directory for the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=None, help="Override fine-tune epoch count (default: cfg epochs // 3)")
    parser.add_argument("--lr", type=float, default=None, help="Override fine-tune learning rate (default: cfg lr / 5)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze all layers except the output heads")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_root = (ROOT.parent / cfg["data"]["raw_root"]).resolve()
    subject_dir = raw_root / args.subject

    out_dir = Path(args.out) if args.out else ROOT / "models" / args.subject / "finetuned"
    result = run_finetune(
        subject_dir, cfg, args.pretrained, out_dir,
        epochs=args.epochs, lr=args.lr, freeze_backbone=args.freeze_backbone,
    )

    print(f"\nFine-tuned model saved -> {out_dir}")
    print("Test-set forecast metrics:")
    for k, v in result["test_metrics"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
