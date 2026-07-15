"""Minimal Transformer time-series forecaster (benchmark for long-context / 15+ day data)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    """Input: (batch, seq_len, n_features). Output: mean & std for each horizon.

    Uses a causal attention mask so predictions only depend on past/current steps.
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mean_head = nn.Linear(d_model, n_horizons)
        self.std_head = nn.Linear(d_model, n_horizons)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        h = self.encoder(h, mask=causal_mask, is_causal=True)
        last = h[:, -1, :]
        mean = self.mean_head(last)
        std = self.softplus(self.std_head(last)) + 1e-3
        return mean, std
