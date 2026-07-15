"""First-class anomaly detector built on UNCERTAINTY-NORMALIZED residuals.

This is the rebuilt anomaly layer (Tier-1 items 2-5). It is deliberately kept
separate from the forecasting model: it takes the model's per-timestep outputs
(y_true, predicted mean, predicted std) and turns them into an interpretable
anomaly score + alerts, saving every intermediate quantity for debugging.

Pipeline
--------
1. z_t = |y_t - mu_t| / (sigma_t + eps)         per target, clipped       (item 3)
2. combined score = weighted blend of HR z-scores and RMSSD z-scores       (item 3)
3. effective score = combined * signal-quality downweight                  (item 6)
4. alerts via hysteresis (start/stop thresholds) + min-duration + cooldown  (item 4)
5. threshold calibrated on VALIDATION-NORMAL scores (percentile)            (item 5)
6. report normal alert rate + false-alerts-per-hour                        (item 5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- item 3
def uncertainty_zscores(y_true, y_mean, y_std, eps: float = 1e-6, clip: float = 8.0) -> np.ndarray:
    """z = |y - mu| / (sigma + eps), per target, clipped to avoid one artifact
    exploding the score. Inputs are (N, T); returns (N, T)."""
    y_true = np.asarray(y_true, float); y_mean = np.asarray(y_mean, float)
    y_std = np.asarray(y_std, float)
    z = np.abs(y_true - y_mean) / (np.clip(y_std, 0.0, None) + eps)
    return np.clip(z, 0.0, clip)


def _target_weight(name: str, w_hr: float, w_rmssd: float, horizon_decay: float) -> float:
    grp = "RMSSD" if name.upper().startswith("RMSSD") else "HR"
    base = w_rmssd if grp == "RMSSD" else w_hr
    # shorter horizons are more actionable -> weight them slightly higher
    hor = 0
    if "_target_" in name:
        try:
            hor = int(name.split("_target_")[1].rstrip("s"))
        except ValueError:
            hor = 0
    decay = horizon_decay ** (hor / 60.0) if hor else 1.0
    return base * decay


def combine_scores(z: np.ndarray, target_cols: list[str],
                   w_hr: float = 1.0, w_rmssd: float = 0.7,
                   horizon_decay: float = 0.9, extra: np.ndarray | None = None) -> np.ndarray:
    """Weighted blend of per-target z-scores into one score per timestep.

    HR and RMSSD contribute via separate weights so HRV collapse is scored
    distinctly from HR deviation. `extra` (e.g. a baseline-drift score) is added.
    Returns a (N,) score in z-like units (a weighted average of clipped z's).
    """
    z = np.asarray(z, float)
    w = np.array([_target_weight(c, w_hr, w_rmssd, horizon_decay) for c in target_cols], float)
    w = w / (w.sum() + 1e-9)
    score = z @ w
    if extra is not None:
        score = score + np.asarray(extra, float)
    return score


# ----------------------------------------------------------------- item 4
def hysteresis_alerts(score: np.ndarray, start_thresh: float, stop_thresh: float,
                      min_duration_s: int = 20, cooldown_s: int = 60, fs: float = 1.0) -> np.ndarray:
    """Alert state machine with start/stop hysteresis, minimum persistence and a
    post-alert cooldown. Returns a boolean array.

    - An alert STARTS only after `score >= start_thresh` for `min_duration_s`
      continuous seconds (kills flicker / single-sample spikes).
    - Once active it STAYS until `score < stop_thresh` (< start) -> no on/off
      chatter around the threshold.
    - After it ends, no new alert for `cooldown_s` unless the score is very high
      (>= 1.5x start), which bypasses cooldown for genuine emergencies.
    """
    score = np.asarray(score, float)
    n = len(score)
    out = np.zeros(n, dtype=bool)
    min_dur = max(1, int(round(min_duration_s * fs)))
    cooldown = int(round(cooldown_s * fs))

    state = "quiet"
    run = 0
    cool = 0
    for i in range(n):
        s = score[i]
        if state == "quiet":
            if cool > 0:
                cool -= 1
            bypass = s >= 1.5 * start_thresh
            if s >= start_thresh and (cool == 0 or bypass):
                run += 1
                if run >= min_dur:
                    state = "active"
                    out[max(0, i - run + 1): i + 1] = True   # backfill the build-up
            else:
                run = 0
        else:  # active
            out[i] = True
            if s < stop_thresh:
                state = "quiet"
                run = 0
                cool = cooldown
    return out


# ----------------------------------------------------------------- item 5
def calibrate_threshold(normal_scores: np.ndarray, percentile: float = 99.0,
                        min_value: float = 1.0) -> float:
    """Pick the alert START threshold from NORMAL (non-anomalous) validation
    scores: the given percentile of the normal-score distribution. This targets
    a false-alert budget instead of eyeballing a threshold."""
    s = np.asarray(normal_scores, float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return max(min_value, 3.0)
    return float(max(min_value, np.percentile(s, percentile)))


def alert_events(alerts: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx_exclusive), ...] contiguous alert runs."""
    alerts = np.asarray(alerts, bool)
    events = []
    i = 0
    n = len(alerts)
    while i < n:
        if alerts[i]:
            j = i
            while j < n and alerts[j]:
                j += 1
            events.append((i, j))
            i = j
        else:
            i += 1
    return events


def alert_stats(alerts: np.ndarray, fs: float = 1.0) -> dict:
    """Normal alert rate, false-alerts-per-hour, count, mean duration."""
    alerts = np.asarray(alerts, bool)
    n = len(alerts)
    events = alert_events(alerts)
    hours = n / fs / 3600.0 if n else 0.0
    durations = [(b - a) / fs for a, b in events]
    return {
        "alert_rate": float(alerts.mean()) if n else 0.0,          # fraction of time in alert
        "n_events": len(events),
        "alerts_per_hour": (len(events) / hours) if hours > 0 else 0.0,
        "mean_alert_duration_s": float(np.mean(durations)) if durations else 0.0,
        "total_alert_time_s": float(np.sum(durations)) if durations else 0.0,
    }


# ----------------------------------------------------------------- item 2
def build_score_frame(index, y_true, y_mean, y_std, target_cols: list[str],
                      sqi_overall: np.ndarray | None = None,
                      start_thresh: float = 3.0, stop_thresh: float | None = None,
                      min_duration_s: int = 20, cooldown_s: int = 60, fs: float = 1.0,
                      w_hr: float = 1.0, w_rmssd: float = 0.7) -> pd.DataFrame:
    """Assemble the full per-timestep anomaly record (the debuggable artifact).

    Saves for every timestep: predicted mean/std, residual, normalized residual
    (z) per target, the combined score, the SQI-downweighted effective score,
    and the final alert flag. Returns a DataFrame indexed by `index`.
    """
    from . import signal_quality  # noqa: PLC0415

    y_true = np.asarray(y_true, float); y_mean = np.asarray(y_mean, float); y_std = np.asarray(y_std, float)
    resid = y_true - y_mean
    z = uncertainty_zscores(y_true, y_mean, y_std)
    combined = combine_scores(z, target_cols, w_hr=w_hr, w_rmssd=w_rmssd)

    if sqi_overall is not None:
        dw = signal_quality.quality_downweight(sqi_overall)
        effective = combined * dw
    else:
        dw = np.ones(len(combined))
        effective = combined

    stop_thresh = stop_thresh if stop_thresh is not None else 0.6 * start_thresh
    alerts = hysteresis_alerts(effective, start_thresh, stop_thresh,
                               min_duration_s=min_duration_s, cooldown_s=cooldown_s, fs=fs)

    out = pd.DataFrame(index=index)
    for j, c in enumerate(target_cols):
        out[f"mean_{c}"] = y_mean[:, j]
        out[f"std_{c}"] = y_std[:, j]
        out[f"residual_{c}"] = resid[:, j]
        out[f"znorm_{c}"] = z[:, j]
    out["score_combined"] = combined
    out["sqi"] = dw
    out["score_effective"] = effective
    out["alert"] = alerts
    return out
