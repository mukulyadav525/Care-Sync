"""Torch-specific dataset wrapper — kept separate so datasets.py stays torch-free.

Importing datasets.py for XGBoost paths must not initialise the MPS/CUDA
backend; putting SequenceDataset here lets us lazy-import this module only
in the non-XGBoost (sequence model) training path.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Wraps pre-windowed arrays for use with a torch DataLoader."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
