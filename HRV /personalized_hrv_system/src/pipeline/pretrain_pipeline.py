"""Global (population) pretraining across multiple subjects.

Stage 1 of population-to-personal transfer learning (DESIGN.md addendum):
train one model on pooled windows from many subjects so it learns generic
HR/HRV dynamics, then `finetune_pipeline.run_finetune` adapts it to one
person's data.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from ..features import build_features
from ..models import datasets, train
from ..models.tcn import TCNForecaster
from ..models.torch_datasets import SequenceDataset
from .train_pipeline import load_and_featurize


def run_pretraining(subject_dirs: list[Path | str], cfg: dict, out_dir: Path | str, model_type: str = "tcn") -> dict:
    """Train a global model on pooled windows from all subjects in `subject_dirs`.

    Feature columns are intersected across subjects (in case some subjects'
    feature tables are missing a column, e.g. no RMSSD), so the resulting
    `feature_cols` may be a subset of any single subject's full feature set.
    Saves artifacts to `out_dir` in the same format as `train_pipeline.run_training`.
    """
    horizons = cfg["model"]["horizons_s"]
    seq_len = cfg["model"]["input_seq_len_s"]
    stride = cfg["model"]["stride_s"]

    per_subject = []  # list of (X_train, X_val, y_train, y_val, cols)
    feature_cols: list[str] | None = None
    target_cols: list[str] | None = None

    for subject_dir in subject_dirs:
        table, _ = load_and_featurize(subject_dir, cfg)
        cols = build_features.numeric_feature_columns(table, cfg)
        tcols = build_features.build_target_cols(table, cfg)
        cols = [c for c in cols if c not in tcols]

        if target_cols is None:
            target_cols = tcols
        elif tcols != target_cols:
            raise ValueError(f"Subject {subject_dir} has different target columns ({tcols}) than the rest ({target_cols}).")

        if feature_cols is None:
            feature_cols = cols
        else:
            cols_set = set(cols)
            feature_cols = [c for c in feature_cols if c in cols_set]

        X, y, _ = datasets.make_windows(table, cols, tcols, seq_len, stride)
        if len(X) < 20:
            continue
        train_sl, val_sl, _ = datasets.chronological_split(len(X), cfg["model"]["val_fraction"], cfg["model"]["test_fraction"])
        per_subject.append((X[train_sl], X[val_sl], y[train_sl], y[val_sl], cols))

    if feature_cols is None or not per_subject:
        raise ValueError("No usable subjects found for pretraining.")

    def align(X: np.ndarray, cols: list[str]) -> np.ndarray:
        idx = [cols.index(c) for c in feature_cols]
        return X[:, :, idx]

    X_train = np.concatenate([align(x, cols) for x, _, _, _, cols in per_subject], axis=0)
    X_val = np.concatenate([align(x, cols) for _, x, _, _, cols in per_subject], axis=0)
    y_train = np.concatenate([y for _, _, y, _, _ in per_subject], axis=0)
    y_val = np.concatenate([y for _, _, _, y, _ in per_subject], axis=0)

    scaler = datasets.fit_scaler(X_train)
    X_train_s = datasets.apply_scaler(X_train, scaler)
    X_val_s = datasets.apply_scaler(X_val, scaler)

    # Normalise targets so the model works in unit-variance space.  This is the
    # same fix applied to train_pipeline.run_training — without it the model
    # predicts normalized values (~0) but targets are raw bpm, causing the
    # ~10-20 bpm prediction / ~60 bpm residual artefact at inference.
    target_scaler = datasets.fit_target_scaler(y_train)
    y_train_s = datasets.apply_target_scaler(y_train, target_scaler)
    y_val_s = datasets.apply_target_scaler(y_val, target_scaler)

    train_ds = SequenceDataset(X_train_s, y_train_s)
    val_ds = SequenceDataset(X_val_s, y_val_s)

    n_features = len(feature_cols)
    n_horizons = len(target_cols)

    if model_type == "tcn":
        model_kwargs = {
            "n_features": n_features,
            "n_horizons": n_horizons,
            "hidden_channels": cfg["model"]["hidden_channels"],
            "levels": cfg["model"]["tcn_levels"],
            "kernel_size": cfg["model"]["kernel_size"],
            "dropout": cfg["model"]["dropout"],
        }
        model = TCNForecaster(**model_kwargs)
    elif model_type in ("lstm", "gru"):
        from ..models.lstm_gru import GRUForecaster, LSTMForecaster

        # Use hidden_channels as the canonical key (same as TCN) so load_model
        # can map it to hidden_size uniformly.
        model_kwargs = {
            "n_features": n_features,
            "n_horizons": n_horizons,
            "hidden_channels": cfg["model"]["hidden_channels"],
        }
        ctor = LSTMForecaster if model_type == "lstm" else GRUForecaster
        model = ctor(n_features=n_features, n_horizons=n_horizons, hidden_size=cfg["model"]["hidden_channels"])
    elif model_type == "transformer":
        from ..models.transformer import TransformerForecaster

        model_kwargs = {
            "n_features": n_features,
            "n_horizons": n_horizons,
            "hidden_channels": cfg["model"]["hidden_channels"],
        }
        model = TransformerForecaster(n_features=n_features, n_horizons=n_horizons, d_model=cfg["model"]["hidden_channels"])
    else:
        raise ValueError(f"Unsupported model_type for pretraining: {model_type}")

    history = train.train_model(
        model, train_ds, val_ds,
        epochs=cfg["model"]["epochs"], lr=cfg["model"]["lr"], batch_size=cfg["model"]["batch_size"],
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
            "model_kwargs": model_kwargs,
            "n_subjects": len(per_subject),
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
        "n_subjects": len(per_subject),
    }
