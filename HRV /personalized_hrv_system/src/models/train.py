"""Training loop shared by TCN / LSTM / GRU / Transformer forecasters."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .tcn import gaussian_nll_loss


def train_model(
    model: torch.nn.Module,
    train_ds,
    val_ds,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str | None = None,
    verbose: bool = True,
    beta: float = 0.5,
) -> dict:
    """Train `model` with beta-NLL loss. Returns history dict and leaves
    `model` holding the best (lowest val loss) weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            mean, std = model(X)
            loss = gaussian_nll_loss(mean, std, y, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    mean, std = model(X)
                    # select the best model by MEAN-tracking error (MSE), not NLL,
                    # so a variance-collapsed model can never look "best".
                    val_losses.append(torch.mean((mean - y) ** 2).item())
            val_loss = float(np.mean(val_losses))
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            val_loss = train_loss

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"epoch {epoch+1:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run inference, returns (mean, std) arrays of shape (N, n_horizons)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    X_t = torch.from_numpy(X).to(device)
    mean, std = model(X_t)
    return mean.cpu().numpy(), std.cpu().numpy()


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, MAPE per horizon (columns of y_true/y_pred)."""
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None)), axis=0) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
