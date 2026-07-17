"""Windowing of the feature table into sequence -> multi-horizon-target samples."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_windows(
    table: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window of length `seq_len` over `table`, producing:
      X: (N, seq_len, F) input sequences ending at index i (inclusive)
      y: (N, H) targets at index i (the `HR_target_*` columns, already future-shifted)
      end_idx: (N,) integer positions (into `table`) of the last input timestep
    Rows with any NaN in features or targets within the window are skipped.
    """
    feats = table[feature_cols].to_numpy(dtype=np.float32)
    targets = table[target_cols].to_numpy(dtype=np.float32)

    n = len(table)
    X, y, end_idx = [], [], []
    for end in range(seq_len - 1, n, stride):
        start = end - seq_len + 1
        x_win = feats[start: end + 1]
        y_win = targets[end]
        if np.isnan(x_win).any() or np.isnan(y_win).any():
            continue
        X.append(x_win)
        y.append(y_win)
        end_idx.append(end)

    if not X:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty((0, len(target_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return np.stack(X), np.stack(y), np.array(end_idx, dtype=np.int64)


def chronological_split(n: int, val_fraction: float, test_fraction: float) -> tuple[slice, slice, slice]:
    """Split `n` samples chronologically into train/val/test slices."""
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test
    train = slice(0, n_train)
    val = slice(n_train, n_train + n_val)
    test = slice(n_train + n_val, n)
    return train, val, test


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler over the flattened (N*L, F) training windows."""
    n, l, f = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(n * l, f))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, l, f = X.shape
    flat = scaler.transform(X.reshape(n * l, f))
    return flat.reshape(n, l, f).astype(np.float32)


def fit_target_scaler(y_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler over training targets (N, H).

    Normalizing targets alongside features stabilises Gaussian-NLL optimisation
    when targets span very different ranges (HR in bpm, RMSSD in ms).
    """
    scaler = StandardScaler()
    scaler.fit(y_train)
    return scaler


def apply_target_scaler(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(y).astype(np.float32)


def inverse_target_scaler(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse-transform predicted means back to the original target units."""
    return scaler.inverse_transform(y_scaled)


def inverse_target_std(y_std_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Std lives in scaled space; multiply by the per-target std to go back to raw units."""
    return y_std_scaled * scaler.scale_
