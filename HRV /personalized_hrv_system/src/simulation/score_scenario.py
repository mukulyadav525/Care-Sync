"""Score a simulated scenario with the Tier-1/2 anomaly detector.

Shared by `scripts/run_simulation.py` (scorecard table) and
`scripts/plot_simulation.py` (plots) so both use the SAME, improved logic:

  persistence forecast -> adaptive (causal) residual std -> uncertainty z-scores
  -> SQI down-weight -> threshold calibrated on the scenario's NORMAL windows
  -> hysteresis alerts -> ROC/PR/event scorecard.

Calibrating the threshold on label==0 (normal) windows — which include benign
exercise blocks — is what stops normal activity from constantly alerting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..anomaly import anomaly_eval, detector, scoring, signal_quality
from ..anomaly.explain import illness_score


def _persistence(df: pd.DataFrame, signals, horizons):
    """Observation-time-aligned persistence: at time t, the 'prediction' is the
    value h seconds ago (s[t-h]) and the actual is s[t]. This puts the anomaly
    signal AT the anomaly (no h-second lead), so short events align with labels."""
    n = len(df)
    cols, yts, yps = [], [], []
    for sig in signals:
        if sig not in df.columns:
            continue
        s = df[sig].to_numpy(dtype=float)
        for h in horizons:
            yt = np.full(n, np.nan); yp = np.full(n, np.nan)
            yt[h:] = s[h:]
            yp[h:] = s[:-h]
            cols.append(f"{sig}_target_{h}s"); yts.append(yt); yps.append(yp)
    return cols, np.stack(yts, 1), np.stack(yps, 1)


def _baseline_deviation(df: pd.DataFrame, valid: np.ndarray) -> np.ndarray:
    """Illness-like sustained-deviation term: HR elevated AT REST + RMSSD drop +
    TEMP rise, measured vs a causal EXPANDING baseline (holds the early-healthy
    reference instead of adapting away). Catches slow/sustained anomalies that
    forecast residuals miss."""
    hr = pd.Series(df["HR"].to_numpy()[valid])
    bm, bs = hr.expanding(min_periods=300).mean(), hr.expanding(min_periods=300).std()
    hr_dev = ((hr - bm) / bs.replace(0, np.nan)).clip(lower=0).fillna(0.0).to_numpy()
    act = df["activity_intensity"].to_numpy()[valid] if "activity_intensity" in df.columns else np.zeros(valid.sum())
    hr_dev = hr_dev * (act < 0.3)                       # only suspicious when still

    if "RMSSD" in df.columns:
        rm = pd.Series(df["RMSSD"].to_numpy()[valid])
        rbase = rm.expanding(min_periods=300).mean()
        rmssd_drop = ((rbase - rm) / rbase.replace(0, np.nan)).clip(lower=0).fillna(0.0).to_numpy()
    else:
        rmssd_drop = np.zeros(valid.sum())

    if "TEMP" in df.columns:
        tm = pd.Series(df["TEMP"].to_numpy()[valid])
        temp_dev = (tm - tm.expanding(min_periods=300).mean()).clip(lower=0).fillna(0.0).to_numpy()
    else:
        temp_dev = np.zeros(valid.sum())

    return illness_score(hr_dev, rmssd_drop, temp_dev)   # ~1.0 for a clear illness


def _adaptive_std(resid: np.ndarray, window: int = 120) -> np.ndarray:
    """Causal rolling std of residuals per column; early/empty -> column median prior."""
    rdf = pd.DataFrame(resid)
    std = rdf.rolling(window, min_periods=10).std()
    std = std.ffill()                                  # causal carry-forward
    for j in range(std.shape[1]):
        col = std.iloc[:, j]
        prior = np.nanmedian(col.to_numpy())
        std.iloc[:, j] = col.fillna(prior if np.isfinite(prior) else 2.0)
    return np.clip(std.to_numpy(), 1e-3, None)


def score_scenario(df: pd.DataFrame, cfg: dict, horizons) -> dict:
    acfg = cfg["anomaly"]
    df = df.copy()
    df["HR"] = df["HR"].interpolate()
    signals = ["HR"] + (["RMSSD"] if "RMSSD" in df.columns else [])

    cols, y_true, y_pred = _persistence(df, signals, horizons)
    valid = ~np.isnan(y_true).any(axis=1)
    yt, yp = y_true[valid], y_pred[valid]
    resid = yt - yp
    std = _adaptive_std(resid)

    znorm = detector.uncertainty_zscores(yt, yp, std)
    # baseline-deviation term (illness/sustained), scaled into z-units so a clear
    # illness contributes ~4 to the score (above a normal-calibrated threshold).
    extra = _baseline_deviation(df, valid) * acfg.get("baseline_weight", 4.0)
    # CUSUM drift channel on the longest-horizon residual: accumulates evidence
    # for slow sustained directional drift that per-step z-scores normalise away.
    hr_long = cols.index(f"HR_target_{horizons[-1]}s") if f"HR_target_{horizons[-1]}s" in cols else 0
    cusum = scoring.cusum_drift_score(resid[:, hr_long])
    extra = extra + np.minimum(cusum, 8.0) * acfg.get("cusum_weight", 0.5)
    combined = detector.combine_scores(znorm, cols, w_hr=acfg.get("w_hr", 1.0),
                                       w_rmssd=acfg.get("w_rmssd", 0.7), extra=extra)

    sqi_overall = signal_quality.compute_signal_quality(df)["sqi_overall"].to_numpy()[valid]
    effective = combined * signal_quality.quality_downweight(sqi_overall) if acfg.get("sqi_downweight", True) else combined

    labels = df["anomaly_label"].to_numpy()[valid].astype(int) if "anomaly_label" in df.columns else np.zeros(valid.sum(), int)
    normal_scores = effective[labels == 0] if (labels == 0).any() else effective
    start = detector.calibrate_threshold(normal_scores, percentile=acfg.get("start_percentile", 99.0),
                                         min_value=acfg.get("min_start_thresh", 3.0))
    stop = acfg.get("stop_fraction", 0.6) * start
    alert = detector.hysteresis_alerts(effective, start, stop,
                                       min_duration_s=acfg.get("min_duration_s", 20),
                                       cooldown_s=acfg.get("cooldown_s", 60), fs=1.0)

    return {
        "t": df.index[valid],
        "hr": df["HR"].to_numpy()[valid],
        "y_pred_1min": yp[:, 0],
        "score": effective,
        "alert": alert,
        "labels": labels,
        "threshold": start,
        "valid": valid,
        "df": df,
        "scorecard": anomaly_eval.scenario_scorecard(effective, alert, labels, fs=1.0),
    }
