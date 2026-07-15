"""Anomaly scoring: residuals, adaptive z-scores, prediction intervals, and the
combined alerting rule (DESIGN.md section 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.kalman import KalmanBaselineTracker
from .adaptive import EWMAStats

# Physiological state codes (mirrors state_classifier.py constants).
# Imported lazily inside functions to avoid circular imports.
_EXERCISE = 4
_RECOVERY = 5


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """r_h(t) = y_h(t) - yhat_h(t)"""
    return y_true - y_pred


def adaptive_zscores(
    resid: np.ndarray,
    activity_buckets: np.ndarray | None = None,
    lam: float = 0.02,
) -> np.ndarray:
    """EWMA-adaptive z-score of residuals, per horizon (columns), optionally
    conditioned on a per-sample activity bucket (DESIGN.md eq. 7.2)."""
    n, h = resid.shape
    z = np.zeros_like(resid)
    stats = [EWMAStats(lam=lam) for _ in range(h)]
    for i in range(n):
        bucket = int(activity_buckets[i]) if activity_buckets is not None and not np.isnan(activity_buckets[i]) else 0
        for j in range(h):
            z[i, j] = stats[j].zscore(float(resid[i, j]), key=bucket)
    return z


def prediction_interval_scores(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> np.ndarray:
    """score_h(t) = |y_h(t) - yhat_h(t)| / sigmahat_h(t)  (DESIGN.md eq. 7.3)"""
    return np.abs(y_true - y_pred_mean) / np.clip(y_pred_std, 1e-6, None)


def prediction_interval_flags(
    y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray, c: float = 1.96
) -> np.ndarray:
    """Boolean flags: True where y_true falls outside [mean - c*std, mean + c*std]."""
    lower = y_pred_mean - c * y_pred_std
    upper = y_pred_mean + c * y_pred_std
    return (y_true < lower) | (y_true > upper)


def rmssd_drop_zscore(rmssd: np.ndarray, activity_buckets: np.ndarray | None = None, lam: float = 0.02) -> np.ndarray:
    """One-sided z-score for HRV (RMSSD) collapse (DESIGN.md section 7.4, Case 4)."""
    n = len(rmssd)
    z = np.zeros(n)
    stats = EWMAStats(lam=lam)
    for i in range(n):
        bucket = int(activity_buckets[i]) if activity_buckets is not None and not np.isnan(activity_buckets[i]) else 0
        z[i] = stats.zscore(float(rmssd[i]), key=bucket)
    return z


def rmssd_pct_drop(rmssd: np.ndarray, activity_buckets: np.ndarray | None = None, lam: float = 0.02) -> np.ndarray:
    """Fractional drop of RMSSD below its EWMA baseline (0 if at/above baseline).
    Used by the explainable-alert layer (e.g. "RMSSD dropped 35%")."""
    n = len(rmssd)
    out = np.zeros(n)
    stats = EWMAStats(lam=lam)
    for i in range(n):
        bucket = int(activity_buckets[i]) if activity_buckets is not None and not np.isnan(activity_buckets[i]) else 0
        mean, _ = stats.update(float(rmssd[i]), key=bucket)
        out[i] = max(0.0, (mean - rmssd[i]) / mean) if mean > 1e-6 else 0.0
    return out


def kalman_innovations(resid_1min: np.ndarray, process_var: float = 1e-3, measurement_var: float = 1.0) -> np.ndarray:
    """Run a Kalman filter over the 1-minute-horizon residual stream and return
    the normalised innovation at each step (catches gradual drift, Case 3).

    process_var is set higher than the original 1e-4 so the filter tracks slow
    physiological drift rather than under-reacting to it.
    """
    tracker = KalmanBaselineTracker(process_var=process_var, measurement_var=measurement_var)
    out = np.zeros(len(resid_1min))
    for i, r in enumerate(resid_1min):
        _, _, innovation_z = tracker.update(float(r))
        out[i] = innovation_z
    return out


def cusum_drift_score(
    residual: np.ndarray,
    k: float = 0.7,
    lam_var: float = 0.02,
    decay: float = 0.998,
) -> np.ndarray:
    """Two-sided decaying CUSUM detector for sustained directional bias in residuals.

    Unlike the Kalman innovation (which is a per-step measure), CUSUM accumulates
    evidence across many steps and catches slow trends that per-step z-scores
    normalise away (e.g. a sustained +1 bpm/min residual during gradual drift).

    Key design choices
    ------------------
    - Reference mean fixed at 0: forecast residuals should be zero-mean under
      normal conditions.  Only the variance is EWMA-tracked to adapt to noise
      level without being fooled by a drifting mean.
    - Geometric decay (0 < decay < 1): prevents the accumulator from carrying
      over large values from exercise/noise bursts into subsequent rest periods.
      With decay=0.998, the CUSUM halves after ~350 steps (~6 min at 1 Hz).

    Parameters
    ----------
    residual : 1-D forecast residual stream (y_true - y_pred), preferably the
               longest available horizon for maximum drift sensitivity.
    k        : slack (allowance) in sigma units — half the minimum detectable
               sustained shift.  k=0.7 → good sensitivity with low false-alarm
               rate for physiological signals.
    lam_var  : EWMA lambda for the *variance* estimator (not mean).
    decay    : geometric forgetting factor applied per step to both s_plus and
               s_minus.  Decay ≈ 0.998 → ~6-min half-life at 1 Hz.

    Returns the CUSUM statistic normalised to sigma units, directly comparable
    to z_thresh.
    """
    n = len(residual)
    var = 100.0   # initial variance (std≈10 bpm); conservative cold start
    s_plus = np.zeros(n)
    s_minus = np.zeros(n)
    sp = sm = 0.0
    for i in range(n):
        r = float(residual[i])
        var = (1 - lam_var) * (var + lam_var * r ** 2)
        std = max(var ** 0.5, 1e-3)
        z_i = r / std
        sp = max(0.0, decay * sp + z_i - k)
        sm = max(0.0, decay * sm - z_i - k)
        s_plus[i] = sp
        s_minus[i] = sm
    return np.maximum(s_plus, s_minus)


def combined_anomaly_score(
    z_scores: np.ndarray,
    rmssd_z: np.ndarray | None = None,
    kalman_innovation: np.ndarray | None = None,
    circadian_z: np.ndarray | None = None,
    illness_scores: np.ndarray | None = None,
    digital_twin_z: np.ndarray | None = None,
    cusum_score: np.ndarray | None = None,
    activity_context: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-channel anomaly score with exercise suppression and illness boost.

    Design changes vs the original max() implementation
    ---------------------------------------------------
    1. Exercise suppression — during EXERCISE/RECOVERY states, HR-based channels
       (forecast z, circadian z, digital-twin z) are down-weighted by 80 % because
       elevated HR is *expected* during exertion.  RMSSD and illness terms are not
       suppressed.
    2. Illness additive boost — the illness composite (HR elevated + RMSSD drop +
       TEMP rise) is added on top of the max-channel score so that multi-channel
       mild anomalies (each below z_thresh individually) can still trigger an alert.
    3. CUSUM channel — accumulated drift score catches gradual HR trends that
       per-step z-scores normalise away.
    4. Digital twin channel — personalised expected-HR deviation, also
       exercise-suppressed.

    The base score is still max() over the remaining channels so that a single
    strong signal (spike, noise burst) triggers immediately without needing a
    weighted-sum recalibration of the threshold.
    """
    n = z_scores.shape[0]

    # Exercise / recovery suppression factor for HR-based channels.
    if activity_context is not None:
        is_active = np.isin(activity_context, [_EXERCISE, _RECOVERY])
        hr_suppress = np.where(is_active, 0.2, 1.0)
    else:
        hr_suppress = np.ones(n)

    # Build channel list for max() base score.
    components: list[np.ndarray] = []

    # Forecast residual z-scores (exercise-suppressed), all horizons.
    components.append(np.abs(z_scores) * hr_suppress[:, None])

    # RMSSD drop: one-sided (drops only) and NOT exercise-suppressed because a
    # sudden RMSSD collapse is meaningful even at rest during exercise recovery.
    if rmssd_z is not None:
        components.append(np.maximum(0.0, -rmssd_z)[:, None])

    # Kalman filter innovation (exercise-suppressed — large residuals during exercise
    # are expected and should not be treated as drift evidence).
    if kalman_innovation is not None:
        components.append((np.abs(kalman_innovation) * hr_suppress)[:, None])

    # CUSUM drift score: exercise-suppressed and capped at 8.0 to prevent unbounded
    # growth.  During vigorous exercise the 10-min residual is large by design
    # (HR rises toward 150+ bpm), so CUSUM must be gated like the other HR terms.
    if cusum_score is not None:
        components.append((np.minimum(cusum_score, 8.0) * hr_suppress)[:, None])

    # Circadian deviation (exercise-suppressed).
    if circadian_z is not None:
        components.append((np.abs(circadian_z) * hr_suppress)[:, None])

    # Personal digital twin z-score (exercise-suppressed).
    if digital_twin_z is not None:
        components.append((np.abs(digital_twin_z) * hr_suppress)[:, None])

    stacked = np.concatenate(components, axis=1)
    combined = stacked.max(axis=1)

    # Illness score is an additive boost so that multi-channel mild anomalies
    # (each individually below the threshold) can jointly push combined above it.
    # Scale factor 1.5 means a "typical fever" (illness_score ≈ 1.0) adds 1.5 to
    # the combined score — enough to push a borderline case over z_thresh=3.0.
    if illness_scores is not None:
        combined = combined + np.clip(illness_scores, 0, None) * 1.5

    return combined


def severity(score: np.ndarray) -> np.ndarray:
    """Map continuous anomaly score to severity labels."""
    sev = np.full(score.shape, "normal", dtype=object)
    sev[score >= 2] = "watch"
    sev[score >= 3] = "alert"
    sev[score >= 5] = "urgent"
    return sev


def debounce_alerts(flags: np.ndarray, k: int) -> np.ndarray:
    """Require >= k consecutive True flags before raising an alert (reduces noise)."""
    out = np.zeros_like(flags, dtype=bool)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run >= k
    return out
