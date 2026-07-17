"""Inference + anomaly scoring pipeline for a trained personal model."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ..anomaly import detector, scoring, signal_quality
from ..models import datasets, train
from ..models.lstm_gru import GRUForecaster, LSTMForecaster
from ..models.tcn import TCNForecaster
from ..models.transformer import TransformerForecaster
from ..models.xgb_model import XGBForecaster
from ..personalization.digital_twin import digital_twin_score


def load_model(out_dir: Path | str):
    out_dir = Path(out_dir)
    meta = joblib.load(out_dir / "meta.joblib")
    scaler = joblib.load(out_dir / "scaler.joblib")

    # Load target scaler if it exists (trained with target normalisation)
    target_scaler_path = out_dir / "target_scaler.joblib"
    meta["_target_scaler"] = joblib.load(target_scaler_path) if target_scaler_path.exists() else None

    kwargs = meta["model_kwargs"]
    model_type = meta["model_type"]
    if model_type == "xgboost":
        model = XGBForecaster(n_targets=kwargs["n_targets"])
        model.load(out_dir / "xgb")
        return model, scaler, meta
    if model_type == "tcn":
        model = TCNForecaster(**kwargs)
    elif model_type == "lstm":
        model = LSTMForecaster(n_features=kwargs["n_features"], n_horizons=kwargs["n_horizons"], hidden_size=kwargs["hidden_channels"])
    elif model_type == "gru":
        model = GRUForecaster(n_features=kwargs["n_features"], n_horizons=kwargs["n_horizons"], hidden_size=kwargs["hidden_channels"])
    elif model_type == "transformer":
        model = TransformerForecaster(n_features=kwargs["n_features"], n_horizons=kwargs["n_horizons"], d_model=kwargs["hidden_channels"])
    else:
        raise ValueError(model_type)

    state = torch.load(out_dir / f"{model_type}_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, meta


def run_inference(table: pd.DataFrame, model, scaler, meta: dict, cfg: dict, twin: dict | None = None,
                  stride: int = 1) -> pd.DataFrame:
    """Run the model over the full table and compute anomaly scores.

    Returns a DataFrame indexed like `table` (only rows with a full input window
    and known targets) containing predictions, residuals, z-scores, and the
    combined anomaly score / severity.

    The combined anomaly score incorporates:
      - EWMA z-scores of forecast residuals (exercise-context suppressed)
      - Kalman-filter innovation for slow drift
      - CUSUM drift score for sustained directional bias
      - RMSSD drop z-score (one-sided, HRV collapse)
      - Circadian deviation z-score
      - Illness composite score (HR elevated + RMSSD drop + TEMP rise)
      - Digital twin z-score when `twin` is provided

    If `twin` is provided, the digital twin z-score is also stored as a separate
    column `digital_twin_score`.
    """
    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]
    seq_len = meta["seq_len"]
    target_scaler = meta.get("_target_scaler")
    # stride=1 scores every available second (default); larger values subsample
    # the scored windows for faster evaluation on long recordings.

    X, y, end_idx = datasets.make_windows(table, feature_cols, target_cols, seq_len, stride)
    if len(X) == 0:
        return pd.DataFrame()

    X_scaled = datasets.apply_scaler(X, scaler)
    if meta["model_type"] == "xgboost":
        y_pred_mean_scaled, y_pred_std_scaled = model.predict(X_scaled[:, 0, :])
    else:
        y_pred_mean_scaled, y_pred_std_scaled = train.predict(model, X_scaled)

    # Inverse-transform predictions to raw bpm/ms units.  Without this step the
    # model outputs normalised values (~0) while y is in raw bpm, causing the
    # ~10-20 bpm prediction / ~60 bpm residual artefact seen in the plots.
    if target_scaler is not None:
        y_pred_mean = datasets.inverse_target_scaler(y_pred_mean_scaled, target_scaler)
        y_pred_std = datasets.inverse_target_std(y_pred_std_scaled, target_scaler)
    else:
        y_pred_mean, y_pred_std = y_pred_mean_scaled, y_pred_std_scaled

    resid = y - y_pred_mean
    acfg = cfg["anomaly"]

    # --- Uncertainty-normalized anomaly score (Tier-1/2 detector) ---
    znorm = detector.uncertainty_zscores(y, y_pred_mean, y_pred_std)

    # Baseline-deviation term (sustained / illness-like anomalies that forecast
    # residuals miss): HR elevated at rest + RMSSD drop + TEMP rise, from the
    # already-engineered subject-relative features.
    def _col(name):
        return table[name].to_numpy()[end_idx] if name in table.columns else np.zeros(len(end_idx))
    hr_term = np.clip(_col("hr_high_while_still"), 0, 4) / 4
    rmssd_term = np.clip(-_col("RMSSD_z_rolling_3600s"), 0, 4) / 4
    temp_term = np.clip(_col("TEMP_minus_baseline_3600s"), 0, 2) / 2
    extra = (hr_term + rmssd_term + temp_term) / 3.0 * acfg.get("baseline_weight", 4.0)

    # CUSUM drift channel on the longest-horizon HR residual (sustained drift)
    n_h = len(cfg["model"]["horizons_s"])
    hr_long = min(n_h - 1, resid.shape[1] - 1)
    extra = extra + np.minimum(scoring.cusum_drift_score(resid[:, hr_long]), 8.0) * acfg.get("cusum_weight", 0.5)

    combined = detector.combine_scores(
        znorm, target_cols, w_hr=acfg.get("w_hr", 1.0), w_rmssd=acfg.get("w_rmssd", 0.7), extra=extra
    )

    # Signal-quality gate: down-weight the score where the data is unreliable so
    # sensor noise is flagged as an artifact, not a health anomaly.
    sqi_overall = None
    if acfg.get("sqi_downweight", True):
        sqi_df = signal_quality.compute_signal_quality(table)
        sqi_overall = sqi_df["sqi_overall"].to_numpy()[end_idx]
        effective = combined * signal_quality.quality_downweight(sqi_overall)
    else:
        effective = combined

    # Calibrate the START threshold on THIS recording's score distribution
    # (assumes anomalies are rare) -> targets a small false-alert budget.
    start_thresh = detector.calibrate_threshold(
        effective, percentile=acfg.get("start_percentile", 99.0),
        min_value=acfg.get("min_start_thresh", 3.0),
    )
    stop_thresh = acfg.get("stop_fraction", 0.6) * start_thresh
    alert = detector.hysteresis_alerts(
        effective, start_thresh, stop_thresh,
        min_duration_s=acfg.get("min_duration_s", 20),
        cooldown_s=acfg.get("cooldown_s", 60), fs=1.0,
    )

    out = pd.DataFrame(index=table.index[end_idx])
    for i, col in enumerate(target_cols):
        if col.startswith("HR_target_"):
            suffix = col[len("HR_target_"):]
        elif col.startswith("RMSSD_target_"):
            suffix = "rmssd_" + col[len("RMSSD_target_"):]
        else:
            suffix = col
        out[f"y_true_{suffix}"] = y[:, i]
        out[f"y_pred_{suffix}"] = y_pred_mean[:, i]
        out[f"y_std_{suffix}"] = y_pred_std[:, i]
        out[f"residual_{suffix}"] = resid[:, i]
        out[f"znorm_{suffix}"] = znorm[:, i]

    if sqi_overall is not None:
        out["sqi"] = sqi_overall
    out["score_raw"] = combined
    out["anomaly_score"] = effective
    out["alert_threshold"] = start_thresh
    out["alert"] = alert

    # Optional personal digital-twin deviation (informational column)
    if twin is not None and "HR" in table.columns:
        hr_now = table["HR"].to_numpy()[end_idx]
        hours = table.index.hour.to_numpy()[end_idx]
        states = (table["physio_state"].to_numpy()[end_idx]
                  if "physio_state" in table.columns else [None] * len(end_idx))
        out["digital_twin_score"] = [
            digital_twin_score(float(hr_now[i]), int(hours[i]), twin,
                               None if states[i] is None else int(states[i]))
            for i in range(len(end_idx))
        ]

    return out
