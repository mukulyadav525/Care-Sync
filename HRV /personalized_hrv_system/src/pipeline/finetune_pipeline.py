"""Per-subject fine-tuning of a globally pretrained model.

Stage 2 of population-to-personal transfer learning (DESIGN.md addendum):
take the weights from `pretrain_pipeline.run_pretraining` and continue
training on one subject's data only, with a smaller learning rate / fewer
epochs (and optionally a frozen backbone) so the model specializes to that
person without forgetting the population-level dynamics it already learned.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import torch

from ..models import datasets, train
from ..models.torch_datasets import SequenceDataset
from .inference_pipeline import load_model
from .train_pipeline import load_and_featurize


def _freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze every parameter except the final output heads (mean/std), so
    fine-tuning only adapts the last layer(s) to this person's data."""
    head_keywords = ("mean_head", "std_head", "fc_out", "output_layer")
    for name, param in model.named_parameters():
        if any(k in name for k in head_keywords):
            continue
        param.requires_grad = False


def run_finetune(
    subject_dir: Path | str,
    cfg: dict,
    pretrained_dir: Path | str,
    out_dir: Path | str,
    epochs: int | None = None,
    lr: float | None = None,
    freeze_backbone: bool = False,
) -> dict:
    """Fine-tune the model in `pretrained_dir` on `subject_dir`'s data, saving
    the result to `out_dir` (loadable like any personal model via
    `inference_pipeline.load_model`)."""
    pretrained_dir = Path(pretrained_dir)
    model, scaler, meta = load_model(pretrained_dir)

    if meta["model_type"] == "xgboost":
        raise ValueError("Fine-tuning is only supported for the sequence models (tcn/lstm/gru/transformer).")

    # Retrieve target scaler from meta (injected by load_model) so fine-tuning
    # uses the same normalised target space as the pretrained model.
    target_scaler = meta.get("_target_scaler")

    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]
    seq_len = meta["seq_len"]
    stride = cfg["model"]["stride_s"]

    table, _ = load_and_featurize(subject_dir, cfg)
    missing = [c for c in feature_cols if c not in table.columns]
    if missing:
        raise ValueError(f"Subject is missing features required by the pretrained model: {missing}")

    X, y, end_idx = datasets.make_windows(table, feature_cols, target_cols, seq_len, stride)
    if len(X) < 20:
        raise ValueError(f"Not enough windows ({len(X)}) for fine-tuning.")

    train_sl, val_sl, test_sl = datasets.chronological_split(
        len(X), cfg["model"]["val_fraction"], cfg["model"]["test_fraction"]
    )

    # Re-use the global scaler (population feature statistics) so the
    # pretrained weights remain meaningful, rather than refitting from scratch.
    X_scaled = datasets.apply_scaler(X, scaler)

    # Fit a fresh target scaler on this subject if the pretrained model doesn't
    # have one (e.g. an older checkpoint trained before this fix).
    if target_scaler is None:
        target_scaler = datasets.fit_target_scaler(y[train_sl])

    y_scaled = datasets.apply_target_scaler(y, target_scaler)

    train_ds = SequenceDataset(X_scaled[train_sl], y_scaled[train_sl])
    val_ds = SequenceDataset(X_scaled[val_sl], y_scaled[val_sl])

    if freeze_backbone:
        _freeze_backbone(model)

    ft_epochs = epochs if epochs is not None else max(1, cfg["model"]["epochs"] // 3)
    ft_lr = lr if lr is not None else cfg["model"]["lr"] / 5.0

    history = train.train_model(model, train_ds, val_ds, epochs=ft_epochs, lr=ft_lr, batch_size=cfg["model"]["batch_size"])

    # Evaluate on test set in raw units.
    y_pred_mean_scaled, y_pred_std_scaled = train.predict(model, X_scaled[test_sl])
    y_pred_mean = datasets.inverse_target_scaler(y_pred_mean_scaled, target_scaler)
    y_pred_std = datasets.inverse_target_std(y_pred_std_scaled, target_scaler)
    metrics = train.evaluate_forecast(y[test_sl], y_pred_mean)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{meta['model_type']}_model.pt")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(target_scaler, out_dir / "target_scaler.joblib")
    # Strip private runtime keys (e.g. _target_scaler) before persisting meta.
    clean_meta = {k: v for k, v in meta.items() if not k.startswith("_")}
    joblib.dump({**clean_meta, "fine_tuned_from": str(pretrained_dir)}, out_dir / "meta.joblib")

    return {
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "history": history,
        "test_metrics": metrics,
        "test": (X_scaled[test_sl], y[test_sl], y_pred_mean, y_pred_std, end_idx[test_sl]),
        "table": table,
    }
