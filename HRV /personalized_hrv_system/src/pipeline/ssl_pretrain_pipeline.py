"""Self-supervised masked-reconstruction pretraining (Tier-3 #18).

Labelled anomaly data is scarce, but unlabelled wearable windows are abundant.
This pretrains a TCN backbone to RECONSTRUCT randomly-masked feature values
across pooled windows from many subjects (no targets needed). The learned
backbone (`model.tcn`) can then warm-start a forecaster via
`train_pipeline.run_training(..., ssl_init_dir=...)`, giving a better starting
representation than random init — useful for subjects with little data.

The SSL encoder deliberately reuses `tcn.CausalConvBlock` with the SAME structure
as `TCNForecaster.tcn`, so the saved `ssl_encoder.pt` state_dict loads directly
into a forecaster's `.tcn` submodule.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from ..features import build_features
from ..models import datasets
from .train_pipeline import load_and_featurize


def run_ssl_pretraining(subject_dirs, cfg: dict, out_dir: Path | str, mask_prob: float = 0.25) -> dict:
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from ..models.tcn import CausalConvBlock  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len = cfg["model"]["input_seq_len_s"]
    stride = cfg["model"]["stride_s"]

    # pool feature windows (no targets needed); intersect feature columns
    feature_cols, blocks = None, []
    for sd in subject_dirs:
        try:
            table, _ = load_and_featurize(sd, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"[ssl] skip {sd}: {exc}"); continue
        cols = build_features.numeric_feature_columns(table, cfg)
        tcols = build_features.build_target_cols(table, cfg)
        cols = [c for c in cols if c not in tcols]
        X, _, _ = datasets.make_windows(table, cols, tcols[:1], seq_len, stride)
        if len(X) < 20:
            continue
        feature_cols = cols if feature_cols is None else [c for c in feature_cols if c in set(cols)]
        blocks.append((X, cols))
    if feature_cols is None or not blocks:
        raise ValueError("No usable subjects for SSL pretraining.")

    idx_for = lambda cols: [cols.index(c) for c in feature_cols]  # noqa: E731
    X = np.concatenate([b[:, :, idx_for(cols)] for b, cols in blocks], axis=0)
    scaler = datasets.fit_scaler(X)
    X = datasets.apply_scaler(X, scaler)
    Xt = torch.tensor(X, dtype=torch.float32)

    class _MaskedReconTCN(nn.Module):
        def __init__(self, n_features, hidden, levels, kernel, dropout):
            super().__init__()
            layers, in_ch = [], n_features
            for level in range(levels):
                layers.append(CausalConvBlock(in_ch, hidden, kernel, 2 ** level, dropout))
                in_ch = hidden
            self.tcn = nn.Sequential(*layers)
            self.recon = nn.Conv1d(hidden, n_features, 1)

        def forward(self, x):
            h = self.tcn(x.transpose(1, 2))
            return self.recon(h).transpose(1, 2)

    m = cfg["model"]
    model = _MaskedReconTCN(len(feature_cols), m["hidden_channels"], m["tcn_levels"], m["kernel_size"], m["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=m["lr"])
    bs, epochs = m["batch_size"], m["epochs"]
    g = torch.Generator().manual_seed(0)

    model.train()
    last_loss = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        losses = []
        for i in range(0, len(Xt), bs):
            xb = Xt[perm[i:i + bs]]
            mask = (torch.rand(xb.shape, generator=g) < mask_prob).float()
            pred = model(xb * (1 - mask))
            denom = mask.sum().clamp(min=1.0)
            loss = (((pred - xb) ** 2) * mask).sum() / denom
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        last_loss = float(np.mean(losses))
        print(f"[ssl] epoch {ep+1}/{epochs}  recon_mse={last_loss:.4f}")

    torch.save(model.tcn.state_dict(), out_dir / "ssl_encoder.pt")
    joblib.dump({"feature_cols": feature_cols, "mask_prob": mask_prob}, out_dir / "ssl_meta.joblib")
    return {"feature_cols": feature_cols, "recon_mse": last_loss, "n_windows": len(Xt)}


def load_ssl_encoder(forecaster, ssl_dir: Path | str) -> int:
    """Warm-start a TCNForecaster's `.tcn` backbone from a saved SSL encoder.
    Returns the number of parameter tensors successfully loaded."""
    import torch  # noqa: PLC0415
    ssl_dir = Path(ssl_dir)
    state = torch.load(ssl_dir / "ssl_encoder.pt", map_location="cpu", weights_only=True)
    own = forecaster.tcn.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(matched)
    forecaster.tcn.load_state_dict(own)
    return len(matched)
