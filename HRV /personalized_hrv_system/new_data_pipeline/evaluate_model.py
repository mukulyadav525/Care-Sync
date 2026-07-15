#!/usr/bin/env python3
"""Evaluate a trained forecaster on a NEW (held-out) dataset.

Runs the existing inference pipeline over every subject in a dataset, then reports:

  * REGRESSION quality of the forecast — MAE / RMSE / MAPE / R2 / bias / Pearson r
    and 90/95% prediction-interval coverage, per horizon, for HR (and RMSSD).
  * CLASSIFICATION quality — precision / recall / F1 / specificity / accuracy, per
    horizon, by framing the forecast as an "elevated-HR event" detector (does the
    model anticipate HR crossing into an elevated regime?). This is what gives the
    F1 / recall / precision numbers for a forecasting model.

Nothing is retrained; we only load `--model-dir` (its meta.joblib defines the
features / targets / window length) and score it on the new data.

Usage
-----
    # all subjects in the dataset, with the pooled multi-dataset TCN model:
    python new_data_pipeline/evaluate_model.py \
        --config new_data_pipeline/config_newdata.yaml \
        --model tcn --model-dir models/multi/tcn

    # a specific dataset path + a couple of subjects, absolute-bpm event def:
    python new_data_pipeline/evaluate_model.py \
        --model tcn --model-dir models/global/tcn \
        --raw-root /home/me/NEW_DATASET --subjects S2 S5 \
        --event-mode absolute --event-bpm 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from eval_utils import (ROOT, align_table_to_model, discover_subjects, event_labels,
                        load_and_featurize, load_model, resolve_raw_root)

from src.evaluation import metrics as eval_metrics
from src.pipeline.inference_pipeline import run_inference


def _suffix(col: str) -> str:
    """Match the column suffix scheme used by run_inference's output frame."""
    if col.startswith("HR_target_"):
        return col[len("HR_target_"):]
    if col.startswith("RMSSD_target_"):
        return "rmssd_" + col[len("RMSSD_target_"):]
    return col


def _arrays_from_results(df, target_cols):
    """Rebuild (N,T) y_true / y_pred / y_std arrays from a run_inference frame."""
    yt = np.column_stack([df[f"y_true_{_suffix(c)}"].to_numpy() for c in target_cols])
    yp = np.column_stack([df[f"y_pred_{_suffix(c)}"].to_numpy() for c in target_cols])
    ys = np.column_stack([df[f"y_std_{_suffix(c)}"].to_numpy() for c in target_cols])
    return yt, yp, ys


def _event_report(y_true_hr, y_pred_hr, horizons, target_cols, mode, k, abs_bpm):
    """Per-horizon precision/recall/F1 for elevated-HR-event detection."""
    out = {}
    for h in horizons:
        col = f"HR_target_{h}s"
        if col not in target_cols:
            continue
        j = target_cols.index(col)
        # Threshold is defined on the TRUE future-HR distribution (person-relative),
        # then both truth and prediction are binarised with the SAME threshold.
        yt_evt, thr = event_labels(y_true_hr[:, j], mode=mode, k=k, abs_bpm=abs_bpm)
        yp_evt, _ = event_labels(y_pred_hr[:, j], mode="absolute", abs_bpm=thr)
        cm = eval_metrics.classification_metrics(yt_evt, yp_evt, labels=[0, 1])
        pos = cm["per_class"].get("1", {})
        out[f"{h}s"] = {
            "threshold_bpm": thr,
            "n_events_true": int(pos.get("support", 0)),
            "n_events_pred": int(pos.get("TP", 0) + pos.get("FP", 0)),
            "precision": pos.get("precision", 0.0),
            "recall": pos.get("recall", 0.0),
            "f1": pos.get("f1", 0.0),
            "specificity": pos.get("specificity", 0.0),
            "accuracy": cm["accuracy"],
            "mcc": cm["mcc"],
            "n": cm["n"],
        }
    return out


def _fmt_events(rep: dict, title: str) -> str:
    lines = [f"=== {title} ==="]
    lines.append(f"  {'horizon':>8} {'thr(bpm)':>9} {'events':>7} {'prec':>6} "
                 f"{'rec':>6} {'F1':>6} {'spec':>6} {'acc':>6} {'MCC':>6}")
    for hor, m in rep.items():
        lines.append(
            f"  {hor:>8} {m['threshold_bpm']:>9.1f} {m['n_events_true']:>7} "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} "
            f"{m['specificity']:>6.3f} {m['accuracy']:>6.3f} {m['mcc']:>6.3f}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "new_data_pipeline" / "config_newdata.yaml"))
    p.add_argument("--model", default="tcn", choices=["tcn", "lstm", "gru", "transformer", "xgboost"])
    p.add_argument("--model-dir", required=True, help="Trained model directory (contains meta.joblib)")
    p.add_argument("--raw-root", default=None, help="Override dataset root (else taken from config)")
    p.add_argument("--subjects", nargs="*", default=None, help="Subset of subjects (default: all found)")
    p.add_argument("--stride", type=int, default=1, help="Score every Nth second (1 = every second)")
    p.add_argument("--event-mode", default="personal_sigma", choices=["personal_sigma", "absolute"])
    p.add_argument("--event-k", type=float, default=1.0, help="k for mean+k*std (personal_sigma mode)")
    p.add_argument("--event-bpm", type=float, default=100.0, help="Threshold bpm (absolute mode)")
    p.add_argument("--out-dir", default=None, help="Where to write reports + per-subject CSVs")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_root = resolve_raw_root(cfg, args.raw_root)
    subjects = args.subjects or discover_subjects(raw_root)
    if not subjects:
        raise SystemExit(f"No subject folders (S*) found under {raw_root}")

    model, scaler, meta = load_model(args.model_dir)
    target_cols = meta["target_cols"]
    horizons = cfg["model"]["horizons_s"]
    hr_cols = [c for c in target_cols if c.startswith("HR_target_")]

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "new_data_pipeline" / "results" / Path(args.model_dir).name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root : {raw_root}")
    print(f"Model        : {args.model}  ({args.model_dir})")
    print(f"Subjects     : {len(subjects)} -> {subjects}")
    print(f"Targets      : {target_cols}")
    print(f"Event def    : {args.event_mode} (k={args.event_k}, bpm={args.event_bpm})\n")

    subj_reg = []                 # (label, y_true(N,T), y_pred(N,T)) for the regression distribution
    all_t, all_p, all_s = [], [], []
    per_subject_report = {}
    for sid in subjects:
        subj_dir = raw_root / sid
        try:
            table, _ = load_and_featurize(subj_dir, cfg)
            table, missing = align_table_to_model(table, scaler, meta)
            df = run_inference(table, model, scaler, meta, cfg, stride=args.stride)
        except Exception as exc:  # noqa: BLE001 — skip a bad subject, keep going
            print(f"  [skip] {sid}: {exc}")
            continue
        if df is None or df.empty:
            print(f"  [skip] {sid}: no scorable windows (recording shorter than the input window)")
            continue

        df.to_csv(out_dir / f"{sid}_{args.model}_inference.csv")
        yt, yp, ys = _arrays_from_results(df, target_cols)
        all_t.append(yt); all_p.append(yp); all_s.append(ys)
        subj_reg.append((sid, yt, yp))

        reg = eval_metrics.forecast_report(yt, yp, ys, target_cols)
        evt = _event_report(yt, yp, horizons, target_cols, args.event_mode, args.event_k, args.event_bpm)
        per_subject_report[sid] = {"n_windows": int(len(df)), "regression": reg, "events": evt,
                                   "missing_features": missing}
        ev60 = next(iter(evt.values()), {})
        print(f"  [ok]   {sid}: {len(df)} windows  "
              f"HR MAE(overall)={reg['overall']['MAE']:.2f}  "
              f"event F1={ev60.get('f1', float('nan')):.3f}")

    if not all_t:
        raise SystemExit("No subject produced scorable windows — check the dataset path / format.")

    YT, YP, YS = np.concatenate(all_t), np.concatenate(all_p), np.concatenate(all_s)
    overall_reg = eval_metrics.forecast_report(YT, YP, YS, target_cols)
    overall_evt = _event_report(YT, YP, horizons, target_cols, args.event_mode, args.event_k, args.event_bpm)
    dist = eval_metrics.per_subject_metrics(subj_reg, target_cols)

    # ---- console summary ----
    print("\n" + eval_metrics.format_forecast_report(
        overall_reg, title=f"OVERALL forecast quality — {args.model} (n={len(subjects)} subjects)"))
    print("\n" + _fmt_events(overall_evt, f"OVERALL elevated-HR event detection — {args.model}"))
    print("\n" + eval_metrics.format_per_subject(dist, title="Per-subject regression distribution"))

    # ---- persisted reports ----
    report = {
        "model": args.model, "model_dir": str(args.model_dir),
        "raw_root": str(raw_root), "subjects": subjects,
        "target_cols": target_cols,
        "event_definition": {"mode": args.event_mode, "k": args.event_k, "abs_bpm": args.event_bpm},
        "overall": {"regression": overall_reg, "events": overall_evt},
        "per_subject_distribution": dist,
        "per_subject": per_subject_report,
    }
    (out_dir / "metrics_report.json").write_text(json.dumps(report, indent=2, default=float))
    txt = "\n\n".join([
        eval_metrics.format_forecast_report(overall_reg, title=f"OVERALL forecast quality — {args.model}"),
        _fmt_events(overall_evt, "OVERALL elevated-HR event detection"),
        eval_metrics.format_per_subject(dist, title="Per-subject regression distribution"),
    ])
    (out_dir / "metrics_summary.txt").write_text(txt + "\n")
    print(f"\nPer-subject CSVs + metrics_report.json + metrics_summary.txt -> {out_dir}")
    print("Tip: visualise any subject with "
          f"`python scripts/plot_inference.py --subject <S> --model {args.model} "
          f"--csv {out_dir}/<S>_{args.model}_inference.csv`")


if __name__ == "__main__":
    main()
