"""XGBoost tabular-feature forecaster (DESIGN.md addendum: model benchmark).

A non-deep-learning baseline. Unlike TCN/LSTM/GRU/Transformer, it consumes a
single feature vector per timestep (the engineered rolling-window features
already encode recent history) rather than a sequence window. One regressor
is trained per target column. Uncertainty (`predicted std`) is approximated
as the constant residual std on the validation set, per target — coarser than
the sequence models' per-sample heteroscedastic estimate, but sufficient for
prediction-interval scoring.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb


class XGBForecaster:
    def __init__(self, n_targets: int, **xgb_params):
        self.n_targets = n_targets
        self.params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            # nthread=1 prevents OpenMP segfaults when XGBoost runs inside a
            # subprocess on macOS ARM64 (Apple Silicon + XGBoost 3.x issue).
            "nthread": 1,
            **xgb_params,
        }
        self.models = [xgb.XGBRegressor(**self.params) for _ in range(n_targets)]
        self.resid_std = np.ones(n_targets, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        for i, model in enumerate(self.models):
            eval_set = [(X_val, y_val[:, i])] if X_val is not None else None
            model.fit(X, y[:, i], eval_set=eval_set, verbose=False)
        if X_val is not None:
            val_pred = self.predict(X_val)[0]
            self.resid_std = np.clip((y_val - val_pred).std(axis=0), 1e-3, None).astype(np.float32)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) of shape (N, n_targets); std is constant per target."""
        mean = np.stack([m.predict(X) for m in self.models], axis=1)
        std = np.tile(self.resid_std, (len(X), 1))
        return mean, std

    def save(self, out_dir: Path | str) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            model.save_model(str(out_dir / f"xgb_target_{i}.json"))
        np.save(out_dir / "resid_std.npy", self.resid_std)

    def load(self, out_dir: Path | str) -> None:
        out_dir = Path(out_dir)
        for i, model in enumerate(self.models):
            model.load_model(str(out_dir / f"xgb_target_{i}.json"))
        self.resid_std = np.load(out_dir / "resid_std.npy")
