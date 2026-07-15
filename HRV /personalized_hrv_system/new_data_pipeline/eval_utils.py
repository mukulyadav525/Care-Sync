"""Shared helpers for evaluating / running a trained forecaster on a NEW dataset.

Everything here is built on top of the existing pipeline code in ``src/`` — we do
NOT retrain anything. We only:

  * load a model directory (``meta.joblib`` is the source of truth for the feature
    columns, target columns and input window length the model expects),
  * featurize a subject from raw Empatica-E4 files the exact same way training did,
  * align the new subject's feature table to the model's expected columns (a new
    dataset may be missing some sensors — those columns are back-filled with the
    scaler's training mean so they contribute ~0 after standardisation, and a
    warning is emitted),
  * turn the multi-horizon HR forecast into a binary "elevated-HR event" so we can
    report precision / recall / F1 alongside the regression metrics.

The model dir can be a global model (``models/global/<m>``), a personalized one
(``models/<S>/finetuned_<m>``) or a pooled multi-dataset model (``models/multi/<m>``)
— all share the same artifact layout (meta/scaler/target_scaler + weights).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # = personalized_hrv_system/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import datasets                       # noqa: E402
from src.pipeline.inference_pipeline import load_model  # noqa: E402
from src.pipeline.train_pipeline import load_and_featurize  # noqa: E402


# --------------------------------------------------------------------------- io

def resolve_raw_root(cfg: dict, override: str | None = None) -> Path:
    """Resolve the dataset root the same way the rest of the pipeline does:
    relative to the PARENT of the project dir (i.e. the ``HRV`` folder), unless
    an absolute path or ``--raw-root`` override is given."""
    raw = override if override is not None else cfg["data"]["raw_root"]
    p = Path(raw)
    return p if p.is_absolute() else (ROOT.parent / p).resolve()


def discover_subjects(raw_root: Path) -> list[str]:
    """Subject folders are directories whose name starts with 'S' (E4 layout)."""
    raw_root = Path(raw_root)
    if not raw_root.exists():
        return []
    return sorted(p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("S"))


# ------------------------------------------------------------- feature alignment

def align_table_to_model(table: pd.DataFrame, scaler, meta: dict) -> tuple[pd.DataFrame, list[str]]:
    """Ensure `table` has every feature column the model expects, in any order.

    A different dataset may not carry all the sensors the model trained on (e.g.
    no EDA/TEMP). Missing feature columns are created and filled with the scaler's
    per-feature training mean, so after standardisation they sit at ~0 (a neutral
    value) instead of crashing the windowing step. Returns (table, missing_cols).
    """
    feature_cols = meta["feature_cols"]
    means = getattr(scaler, "mean_", None)
    missing = [c for c in feature_cols if c not in table.columns]
    if missing:
        warnings.warn(
            f"[align] new dataset is missing {len(missing)} feature(s) the model "
            f"expects; back-filling with the training mean: {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}"
        )
        for c in missing:
            j = feature_cols.index(c)
            fill = float(means[j]) if means is not None and j < len(means) else 0.0
            table[c] = fill
    return table, missing


def predict_scaled_windows(model, X_scaled: np.ndarray, meta: dict):
    """Run the model over scaled windows and return (mean, std) in RAW units.

    Mirrors the inverse-transform logic in ``inference_pipeline.run_inference`` so
    predictions come back in bpm / ms rather than normalised space.
    """
    target_scaler = meta.get("_target_scaler")
    if meta["model_type"] == "xgboost":
        mean_s, std_s = model.predict(X_scaled[:, 0, :])
    else:
        from src.models import train  # noqa: PLC0415  (lazy: keep torch off the xgb path)
        mean_s, std_s = train.predict(model, X_scaled)
    if target_scaler is not None:
        mean = datasets.inverse_target_scaler(mean_s, target_scaler)
        std = datasets.inverse_target_std(std_s, target_scaler)
    else:
        mean, std = mean_s, std_s
    return mean, std


# ----------------------------------------------------------- "predict next N min"

def latest_forecast(table: pd.DataFrame, model, scaler, meta: dict, cfg: dict) -> dict | None:
    """Forecast the immediate future from the most recent clean input window.

    Unlike full evaluation (which needs the future to be KNOWN so it can score),
    this builds the final ``seq_len`` window even when the future is unknown — that
    is exactly the live "predict the next few minutes" use-case. Returns a dict with
    the reference ("now") timestamp and, per target, the predicted value, std and
    the absolute time the prediction is FOR. Returns None if no clean window exists.
    """
    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]
    seq_len = meta["seq_len"]
    horizons = cfg["model"]["horizons_s"]

    feats = table[feature_cols].to_numpy(dtype=np.float32)
    n = len(feats)
    # walk backwards to the most recent window with no NaNs
    end = None
    for e in range(n - 1, seq_len - 2, -1):
        win = feats[e - seq_len + 1: e + 1]
        if not np.isnan(win).any():
            end = e
            break
    if end is None:
        return None

    X = feats[end - seq_len + 1: end + 1][None, ...]          # (1, seq_len, F)
    X_scaled = datasets.apply_scaler(X, scaler)
    mean, std = predict_scaled_windows(model, X_scaled, meta)
    now = table.index[end]

    preds = {}
    for i, col in enumerate(target_cols):
        signal, hor_s = _parse_target_col(col, horizons)
        preds[col] = {
            "signal": signal,
            "horizon_s": hor_s,
            "for_time": (now + pd.Timedelta(seconds=hor_s)).isoformat() if hor_s else None,
            "pred": float(mean[0, i]),
            "std": float(std[0, i]),
            "lo95": float(mean[0, i] - 1.96 * std[0, i]),
            "hi95": float(mean[0, i] + 1.96 * std[0, i]),
        }
    return {"now": now.isoformat(), "n_seconds_seen": int(n), "predictions": preds}


def _parse_target_col(col: str, horizons: list[int]) -> tuple[str, int | None]:
    """'HR_target_60s' -> ('HR', 60); 'RMSSD_target_300s' -> ('RMSSD', 300)."""
    if "_target_" in col:
        sig, hor = col.split("_target_", 1)
        try:
            return sig, int(hor.rstrip("s"))
        except ValueError:
            return sig, None
    return col, None


# ------------------------------------------------------- elevated-HR event labels

def event_labels(hr_values: np.ndarray, mode: str = "personal_sigma",
                 k: float = 1.0, abs_bpm: float = 100.0,
                 ref: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Binarise HR into an 'elevated-HR event' (1) vs normal (0).

    This is what lets us report precision / recall / F1 for a *forecasting* model:
    the model is graded on whether it correctly anticipates HR crossing into an
    elevated regime at each horizon.

    mode = "personal_sigma": threshold = mean + k*std of `ref` (the recording's
                              own true-HR distribution) -> person-relative.
    mode = "absolute":       threshold = abs_bpm (e.g. 100 bpm tachycardia).
    Returns (binary_labels, threshold).
    """
    hr_values = np.asarray(hr_values, dtype=float)
    if mode == "absolute":
        thr = float(abs_bpm)
    else:  # personal_sigma
        base = np.asarray(ref if ref is not None else hr_values, dtype=float)
        base = base[np.isfinite(base)]
        thr = float(base.mean() + k * base.std()) if len(base) else float(abs_bpm)
    return (hr_values > thr).astype(int), thr
