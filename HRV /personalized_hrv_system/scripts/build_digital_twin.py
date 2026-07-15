#!/usr/bin/env python3
"""Build a Personal Digital Twin profile (JSON) for one subject.

Usage:
    python scripts/build_digital_twin.py --subject S01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.personalization.digital_twin import build_digital_twin, save_digital_twin  # noqa: E402
from src.pipeline.train_pipeline import load_and_featurize  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject folder name, e.g. S01")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--out", default=None, help="Output directory for the digital_twin.json file")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_root = (ROOT.parent / cfg["data"]["raw_root"]).resolve()
    subject_dir = raw_root / args.subject

    table, _ = load_and_featurize(subject_dir, cfg)
    twin = build_digital_twin(table, args.subject)

    out_dir = Path(args.out) if args.out else ROOT / "models" / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "digital_twin.json"
    save_digital_twin(twin, out_path)

    print(f"Saved digital twin -> {out_path}")
    summary = {k: v for k, v in twin.items() if k != "circadian_profile"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
