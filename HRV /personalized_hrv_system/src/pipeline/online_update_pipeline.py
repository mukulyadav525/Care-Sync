"""Online / incremental model adaptation (DESIGN.md addendum: online learning).

Re-processes a subject's latest data and continues training the existing
personal model on the most recent windows only, with a small learning rate
and a small number of epochs. The feature scaler is updated incrementally
(`StandardScaler.partial_fit`) so the model tracks slow shifts in someone's
baseline (e.g. improving fitness lowers resting HR over months).

Intended to be run periodically (e.g. nightly/weekly cron) on top of a model
produced by `train_pipeline.run_training` or `finetune_pipeline.run_finetune`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import torch

from ..models import datasets, train
from ..models.torch_datasets import SequenceDataset
from .inference_pipeline import load_model
from .train_pipeline import load_and_featurize


def run_online_update(
    subject_dir: Path | str,
    cfg: dict,
    model_dir: Path | str,
    out_dir: Path | str | None = None,
) -> dict:
    """Incrementally update the model in `model_dir` using the most recent
    portion of `subject_dir`'s data. Writes the updated artifacts to `out_dir`
    (defaults to overwriting `model_dir`)."""
    model_dir = Path(model_dir)
    out_dir = Path(out_dir) if out_dir else model_dir

    model, scaler, meta = load_model(model_dir)
    if meta["model_type"] == "xgboost":
        raise ValueError("Online updates are only supported for the sequence models (tcn/lstm/gru/transformer).")

    # Retrieve the target scaler loaded by load_model so training stays in the
    # same normalised target space as the original model.
    target_scaler = meta.get("_target_scaler")

    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]
    seq_len = meta["seq_len"]
    stride = cfg["model"]["stride_s"]
    online_cfg = cfg["online"]

    table, _ = load_and_featurize(subject_dir, cfg)
    missing = [c for c in feature_cols if c not in table.columns]
    if missing:
        raise ValueError(f"Subject is missing features required by this model: {missing}")

    X, y, end_idx = datasets.make_windows(table, feature_cols, target_cols, seq_len, stride)
    if len(X) < 20:
        raise ValueError(f"Not enough windows ({len(X)}) for an online update.")

    n_recent = max(20, int(len(X) * online_cfg["recent_fraction"]))
    X_recent, y_recent = X[-n_recent:], y[-n_recent:]

    # Incrementally adapt the feature scaler to this person's recent distribution.
    n, l, f = X_recent.shape
    scaler.partial_fit(X_recent.reshape(n * l, f))
    X_scaled = datasets.apply_scaler(X_recent, scaler)

    # Fit or reuse target scaler — update it with recent data so it tracks
    # long-term baseline shifts (e.g. resting HR dropping as fitness improves).
    if target_scaler is None:
        train_sl_tmp, _, _ = datasets.chronological_split(n_recent, online_cfg["val_fraction"], 0.0)
        target_scaler = datasets.fit_target_scaler(y_recent[train_sl_tmp])
    else:
        # Incrementally update target scaler statistics with the recent window.
        train_sl_tmp, _, _ = datasets.chronological_split(n_recent, online_cfg["val_fraction"], 0.0)
        target_scaler.partial_fit(y_recent[train_sl_tmp])

    y_scaled = datasets.apply_target_scaler(y_recent, target_scaler)

    train_sl, val_sl, _ = datasets.chronological_split(n_recent, online_cfg["val_fraction"], 0.0)
    train_ds = SequenceDataset(X_scaled[train_sl], y_scaled[train_sl])
    val_ds = SequenceDataset(X_scaled[val_sl], y_scaled[val_sl])

    lr = cfg["model"]["lr"] * online_cfg["lr_fraction"]
    history = train.train_model(
        model, train_ds, val_ds,
        epochs=online_cfg["epochs"], lr=lr, batch_size=cfg["model"]["batch_size"], verbose=False,
    )

    # Compute validation metrics in raw bpm/ms units.
    y_pred_mean_scaled, y_pred_std_scaled = train.predict(model, X_scaled[val_sl])
    y_pred_mean = datasets.inverse_target_scaler(y_pred_mean_scaled, target_scaler)
    metrics = train.evaluate_forecast(y_recent[val_sl], y_pred_mean)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{meta['model_type']}_model.pt")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(target_scaler, out_dir / "target_scaler.joblib")
    # Strip private runtime keys before persisting.
    clean_meta = {k: v for k, v in meta.items() if not k.startswith("_")}
    joblib.dump(clean_meta, out_dir / "meta.joblib")

    return {
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "history": history,
        "val_metrics": metrics,
        "n_recent_windows": n_recent,
    }
