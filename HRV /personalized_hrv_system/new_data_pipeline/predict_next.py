#!/usr/bin/env python3
"""Forecast the NEXT few minutes of HR (and RMSSD) for one user.

Give it a user's raw Empatica-E4 folder and a trained model; it featurizes the
recording, takes the most recent clean input window, and predicts the configured
horizons ahead (default 1 / 5 / 10 min) with 95% prediction intervals.

This is the "live" use-case: unlike evaluate_model.py it does NOT need the future
to be known — it predicts past the end of the recording.

Usage
-----
    python new_data_pipeline/predict_next.py \
        --config new_data_pipeline/config_newdata.yaml \
        --model tcn --model-dir models/multi/tcn --subject S2

    # point straight at a folder of E4 CSVs (HR.csv, BVP.csv, ...):
    python new_data_pipeline/predict_next.py \
        --model tcn --model-dir models/global/tcn --data /home/me/somebody/E4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from eval_utils import (ROOT, align_table_to_model, latest_forecast,
                        load_and_featurize, load_model, resolve_raw_root)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "new_data_pipeline" / "config_newdata.yaml"))
    p.add_argument("--model", default="tcn", choices=["tcn", "lstm", "gru", "transformer", "xgboost"])
    p.add_argument("--model-dir", required=True, help="Trained model directory (contains meta.joblib)")
    p.add_argument("--subject", default=None, help="Subject folder name under the dataset root")
    p.add_argument("--data", default=None, help="Direct path to a subject's E4 folder (overrides --subject)")
    p.add_argument("--raw-root", default=None, help="Override dataset root (else from config)")
    p.add_argument("--out", default=None, help="Optional path to write the forecast JSON")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.data:
        subj_dir = Path(args.data)
        label = subj_dir.name
    elif args.subject:
        subj_dir = resolve_raw_root(cfg, args.raw_root) / args.subject
        label = args.subject
    else:
        raise SystemExit("Provide --subject (with a dataset root) or --data (a direct E4 folder path).")
    if not subj_dir.exists():
        raise SystemExit(f"Subject folder not found: {subj_dir}")

    model, scaler, meta = load_model(args.model_dir)
    table, _ = load_and_featurize(subj_dir, cfg)
    table, _missing = align_table_to_model(table, scaler, meta)

    fc = latest_forecast(table, model, scaler, meta, cfg)
    if fc is None:
        raise SystemExit("No clean input window available — recording is shorter than the model's "
                         f"input window ({meta['seq_len']}s) or too gappy.")

    print(f"User            : {label}")
    print(f"Model           : {args.model} ({args.model_dir})")
    print(f"Seconds of data : {fc['n_seconds_seen']}")
    print(f"Forecast made at: {fc['now']}  (predicting forward from here)\n")
    print(f"  {'signal':>7} {'horizon':>8} {'forecast':>10} {'±95% interval':>22} {'for time':>22}")
    for col, m in fc["predictions"].items():
        unit = "bpm" if m["signal"] == "HR" else "ms"
        hor = f"+{m['horizon_s']}s"
        interval = f"[{m['lo95']:.1f}, {m['hi95']:.1f}]"
        for_time = (m["for_time"] or "")[:19].replace("T", " ")
        value = f"{m['pred']:.1f} {unit}"
        print(f"  {m['signal']:>7} {hor:>8} {value:>10} {interval:>22} {for_time:>22}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"user": label, "model": args.model,
                                        "model_dir": str(args.model_dir), **fc}, indent=2))
        print(f"\nSaved forecast -> {out_path}")


if __name__ == "__main__":
    main()
