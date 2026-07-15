"""EDA features: tonic level and phasic (SCR) activity."""
from __future__ import annotations

import pandas as pd


def compute_eda_features(grid: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    eda = grid["EDA"]
    out = pd.DataFrame(index=grid.index)

    tonic = eda.rolling(60, min_periods=10, center=False).mean()
    phasic = eda - tonic

    out["EDA_tonic"] = tonic
    out["EDA_phasic"] = phasic
    out["EDA_phasic_activity_60s"] = phasic.abs().rolling(60, min_periods=10).mean()

    return out
