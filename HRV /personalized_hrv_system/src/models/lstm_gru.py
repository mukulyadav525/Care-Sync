"""LSTM / GRU recurrent baselines, sharing the TCN's mean+std output head."""
from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentForecaster(nn.Module):
    """Input: (batch, seq_len, n_features). Output: mean & std for each horizon."""

    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        cell: str = "lstm",
    ):
        super().__init__()
        cell = cell.lower()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mean_head = nn.Linear(hidden_size, n_horizons)
        self.std_head = nn.Linear(hidden_size, n_horizons)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        mean = self.mean_head(last)
        std = self.softplus(self.std_head(last)) + 1e-3
        return mean, std


def LSTMForecaster(n_features: int, n_horizons: int, **kwargs) -> RecurrentForecaster:
    return RecurrentForecaster(n_features, n_horizons, cell="lstm", **kwargs)


def GRUForecaster(n_features: int, n_horizons: int, **kwargs) -> RecurrentForecaster:
    return RecurrentForecaster(n_features, n_horizons, cell="gru", **kwargs)
