"""End-to-end training pipeline: raw files -> features -> windows -> trained model."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from ..data import loader, sync
from ..evaluation import metrics as eval_metrics
from ..features import build_features
from ..models import datasets


def load_and_featurize(subject_dir: Path | str, cfg: dict):
    """Load + featurize one subject from an Empatica E4 export
    (HR, IBI, ACC, BVP, EDA, TEMP). Covers Dataset3 (Stress-Predict) and WESAD
    (whose CSVs are nested in <SID>/<SID>_E4_Data/, auto-resolved by the loader).
    """
    raw = loader.load_subject_raw(subject_dir)
    synced = sync.build_synced_frame(raw, cfg)
    table = build_features.build_feature_table(synced, cfg)
    return table, synced


def run_training(subject_dir: Path | str, cfg: dict, out_dir: Path | str, model_type: str = "tcn",
                 ssl_init_dir: Path | str | None = None):
    """Train a personalized forecaster for one subject and save artifacts to `out_dir`.

    Returns a dict with the trained model, scaler, feature columns, and test metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table, _ = load_and_featurize(subject_dir, cfg)

    feature_cols = build_features.numeric_feature_columns(table, cfg)
    horizons = cfg["model"]["horizons_s"]
    target_cols = build_features.build_target_cols(table, cfg)
    feature_cols = [c for c in feature_cols if c not in target_cols]

    if model_type == "xgboost":
        from ..models.xgb_model import XGBForecaster  # noqa: PLC0415
        return _run_training_xgboost(table, feature_cols, target_cols, horizons, cfg, out_dir, XGBForecaster)

    # Lazy imports so XGBoost path never initialises PyTorch/MPS (avoids segfault
    # on Apple Silicon when OpenMP and MPS fight over GPU resources).
    import torch  # noqa: PLC0415
    from ..models import train  # noqa: PLC0415
    from ..models.tcn import TCNForecaster  # noqa: PLC0415
    from ..models.torch_datasets import SequenceDataset  # noqa: PLC0415

    seq_len = cfg["model"]["input_seq_len_s"]
    stride = cfg["model"]["stride_s"]
    X, y, end_idx = datasets.make_windows(table, feature_cols, target_cols, seq_len, stride)
    if len(X) < 20:
        raise ValueError(f"Not enough windows ({len(X)}) — check data length / seq_len.")

    train_sl, val_sl, test_sl = datasets.chronological_split(
        len(X), cfg["model"]["val_fraction"], cfg["model"]["test_fraction"]
    )

    scaler = datasets.fit_scaler(X[train_sl])
    X_scaled = datasets.apply_scaler(X, scaler)

    # Normalise targets so the model works in a unit-variance space rather than
    # raw bpm/ms, which stabilises Gaussian-NLL when HR and RMSSD targets are mixed.
    target_scaler = datasets.fit_target_scaler(y[train_sl])
    y_scaled = datasets.apply_target_scaler(y, target_scaler)

    train_ds = SequenceDataset(X_scaled[train_sl], y_scaled[train_sl])
    val_ds = SequenceDataset(X_scaled[val_sl], y_scaled[val_sl])
    test_ds = SequenceDataset(X_scaled[test_sl], y_scaled[test_sl])

    n_features = len(feature_cols)
    n_horizons = len(target_cols)

    if model_type == "tcn":
        model = TCNForecaster(
            n_features=n_features,
            n_horizons=n_horizons,
            hidden_channels=cfg["model"]["hidden_channels"],
            levels=cfg["model"]["tcn_levels"],
            kernel_size=cfg["model"]["kernel_size"],
            dropout=cfg["model"]["dropout"],
        )
        # Tier-3 #18: warm-start the backbone from self-supervised pretraining
        if ssl_init_dir is not None:
            from .ssl_pretrain_pipeline import load_ssl_encoder  # noqa: PLC0415
            n_loaded = load_ssl_encoder(model, ssl_init_dir)
            print(f"[train] SSL warm-start: loaded {n_loaded} backbone tensors from {ssl_init_dir}")
    elif model_type in ("lstm", "gru"):
        from ..models.lstm_gru import GRUForecaster, LSTMForecaster

        ctor = LSTMForecaster if model_type == "lstm" else GRUForecaster
        model = ctor(n_features=n_features, n_horizons=n_horizons, hidden_size=cfg["model"]["hidden_channels"])
    elif model_type == "transformer":
        from ..models.transformer import TransformerForecaster

        model = TransformerForecaster(n_features=n_features, n_horizons=n_horizons, d_model=cfg["model"]["hidden_channels"])
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    history = train.train_model(
        model,
        train_ds,
        val_ds,
        epochs=cfg["model"]["epochs"],
        lr=cfg["model"]["lr"],
        batch_size=cfg["model"]["batch_size"],
        beta=cfg["model"].get("nll_beta", 0.5),
    )

    # test-set evaluation — inverse transform to raw units before computing metrics
    y_pred_mean_scaled, y_pred_std_scaled = train.predict(model, X_scaled[test_sl])
    y_pred_mean = datasets.inverse_target_scaler(y_pred_mean_scaled, target_scaler)
    y_pred_std = datasets.inverse_target_std(y_pred_std_scaled, target_scaler)
    metrics = train.evaluate_forecast(y[test_sl], y_pred_mean)
    regression_report = eval_metrics.regression_metrics(y[test_sl], y_pred_mean, horizon_names=target_cols)

    # persist artifacts
    torch.save(model.state_dict(), out_dir / f"{model_type}_model.pt")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(target_scaler, out_dir / "target_scaler.joblib")
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "horizons_s": horizons,
            "seq_len": seq_len,
            "model_type": model_type,
            "model_kwargs": {
                "n_features": n_features,
                "n_horizons": n_horizons,
                "hidden_channels": cfg["model"]["hidden_channels"],
                "levels": cfg["model"]["tcn_levels"],
                "kernel_size": cfg["model"]["kernel_size"],
                "dropout": cfg["model"]["dropout"],
            },
        },
        out_dir / "meta.joblib",
    )

    return {
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "history": history,
        "test_metrics": metrics,
        "regression_report": regression_report,
        "test": (X_scaled[test_sl], y[test_sl], y_pred_mean, y_pred_std, end_idx[test_sl]),
        "table": table,
    }


def _evaluate_xgb(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Pure-NumPy metrics so the XGBoost path never imports torch/MPS."""
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None)), axis=0) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _run_training_xgboost(table, feature_cols, target_cols, horizons, cfg, out_dir: Path, XGBForecaster):
    """XGBoost benchmark: one regressor per target, fed per-timestep feature
    vectors (no sequence window — reuses `make_windows` with seq_len=1 so the
    same NaN-skipping logic applies)."""
    seq_len = 1
    stride = cfg["model"]["stride_s"]
    X, y, end_idx = datasets.make_windows(table, feature_cols, target_cols, seq_len, stride)
    if len(X) < 20:
        raise ValueError(f"Not enough samples ({len(X)}) — check data length.")

    train_sl, val_sl, test_sl = datasets.chronological_split(
        len(X), cfg["model"]["val_fraction"], cfg["model"]["test_fraction"]
    )

    scaler = datasets.fit_scaler(X[train_sl])
    X_scaled = datasets.apply_scaler(X, scaler)
    X_flat = X_scaled[:, 0, :]

    model = XGBForecaster(n_targets=len(target_cols))
    model.fit(X_flat[train_sl], y[train_sl], X_flat[val_sl], y[val_sl])

    y_pred_mean, y_pred_std = model.predict(X_flat[test_sl])
    metrics = _evaluate_xgb(y[test_sl], y_pred_mean)
    regression_report = eval_metrics.regression_metrics(y[test_sl], y_pred_mean, horizon_names=target_cols)

    model.save(out_dir / "xgb")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "horizons_s": horizons,
            "seq_len": seq_len,
            "model_type": "xgboost",
            "model_kwargs": {"n_targets": len(target_cols)},
        },
        out_dir / "meta.joblib",
    )

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "history": None,
        "test_metrics": metrics,
        "regression_report": regression_report,
        "test": (X_flat[test_sl], y[test_sl], y_pred_mean, y_pred_std, end_idx[test_sl]),
        "table": table,
    }
