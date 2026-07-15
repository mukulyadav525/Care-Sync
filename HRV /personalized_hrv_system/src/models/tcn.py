"""Causal dilated-convolution (TCN) forecaster with heteroscedastic uncertainty head.

Recommended primary model: low, fixed latency, small footprint, and easy to
fine-tune per user (see DESIGN.md section 5).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out[:, :, : -self.padding] if self.padding > 0 else out  # causal trim
        out = self.act(out)
        out = self.drop(out)
        return out + self.residual(x)


class TCNForecaster(nn.Module):
    """Input: (batch, seq_len, n_features). Output: mean & std for each horizon."""

    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        hidden_channels: int = 32,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_ch = n_features
        for level in range(levels):
            dilation = 2 ** level
            layers.append(CausalConvBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))
            in_ch = hidden_channels
        self.tcn = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_channels, n_horizons)
        self.std_head = nn.Linear(hidden_channels, n_horizons)
        self.softplus = nn.Softplus()

        # receptive field in timesteps (informational)
        self.receptive_field = 1 + sum(2 * (kernel_size - 1) * (2 ** level) for level in range(levels))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_features) -> conv expects (batch, channels, seq_len)
        h = x.transpose(1, 2)
        h = self.tcn(h)
        last = h[:, :, -1]  # representation at the most recent timestep
        mean = self.mean_head(last)
        std = self.softplus(self.std_head(last)) + 1e-3
        return mean, std


def gaussian_nll_loss(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor,
                      beta: float = 0.5) -> torch.Tensor:
    """beta-NLL (Seitzer et al. 2022) of `target` under N(mean, std^2).

    Plain Gaussian NLL lets a model minimise the loss by INFLATING the variance
    (predicting the mean with huge sigma) because the (target-mean)^2/var term is
    suppressed by 1/var — exactly the "variance collapse" that makes predictions
    flat. beta-NLL multiplies each sample's NLL by stop_grad(var^beta), which
    restores the mean's gradient magnitude so the mean actually tracks the signal.
    beta=0 recovers standard NLL; beta=0.5 is the recommended default; beta=1
    is equivalent to plain MSE on the mean.
    """
    var = std ** 2
    nll = 0.5 * torch.log(2 * torch.pi * var) + 0.5 * (target - mean) ** 2 / var
    if beta > 0:
        nll = nll * var.detach() ** beta
    return nll.mean()
