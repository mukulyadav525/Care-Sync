"""Uncertainty calibration (Tier-2 #14).

The models predict a Gaussian (mean, std) per target. Anomaly scoring divides by
std, so badly-calibrated std makes the whole anomaly layer weak. This module:

  * measures calibration: prediction-interval coverage at several nominal levels,
    a reliability table, and Gaussian negative-log-likelihood (NLL);
  * fixes calibration POST-HOC: fit a single multiplicative variance-scale `s`
    (a.k.a. temperature for the std) on a held-out VALIDATION set so empirical
    coverage matches nominal, plus a variance floor to stop over-confident std~0.

Apply the fitted scale at inference: std_calibrated = max(s * std, floor).
"""
from __future__ import annotations

import numpy as np

_Z = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def gaussian_nll(y_true, y_mean, y_std) -> float:
    """Mean Gaussian negative log-likelihood (lower = better-calibrated + accurate)."""
    y_true = np.asarray(y_true, float); y_mean = np.asarray(y_mean, float)
    std = np.clip(np.asarray(y_std, float), 1e-6, None)
    nll = 0.5 * np.log(2 * np.pi * std ** 2) + (y_true - y_mean) ** 2 / (2 * std ** 2)
    return float(np.mean(nll))


def reliability_table(y_true, y_mean, y_std, levels=(0.5, 0.8, 0.9, 0.95, 0.99)) -> dict:
    """Empirical coverage at each nominal level (well-calibrated => empirical≈nominal)."""
    y_true = np.asarray(y_true, float); y_mean = np.asarray(y_mean, float)
    std = np.clip(np.asarray(y_std, float), 1e-6, None)
    out = {}
    for lv in levels:
        c = _Z.get(lv, 1.96)
        out[lv] = float((np.abs(y_true - y_mean) <= c * std).mean()) if len(y_true) else float("nan")
    return out


def fit_variance_scale(y_true, y_mean, y_std, target_level: float = 0.95) -> float:
    """Find scalar `s` so that `s*std` gives empirical coverage == target_level.

    Closed-form-ish: at the desired level, the c*std band should contain
    `target_level` of points. We pick `s` = (empirical quantile of |resid|/std at
    target_level) / z(target_level). Robust and one-shot (no optimisation loop).
    """
    y_true = np.asarray(y_true, float); y_mean = np.asarray(y_mean, float)
    std = np.clip(np.asarray(y_std, float), 1e-6, None)
    if len(y_true) == 0:
        return 1.0
    norm_resid = np.abs(y_true - y_mean) / std
    q = np.quantile(norm_resid, target_level)
    z = _Z.get(target_level, 1.96)
    return float(max(q / z, 1e-3))


def calibrate(y_std, scale: float, floor: float = 0.0) -> np.ndarray:
    """Apply a fitted variance scale + floor to predicted std."""
    return np.clip(scale * np.asarray(y_std, float), floor, None)


def calibration_report(y_true, y_mean, y_std, val_true=None, val_mean=None, val_std=None) -> dict:
    """Full report; if validation arrays are given, also fit + apply a scale and
    report calibrated NLL/coverage."""
    rep = {
        "nll": gaussian_nll(y_true, y_mean, y_std),
        "coverage": reliability_table(y_true, y_mean, y_std),
    }
    if val_true is not None:
        scale = fit_variance_scale(val_true, val_mean, val_std, target_level=0.95)
        cal_std = calibrate(y_std, scale)
        rep["fitted_scale"] = scale
        rep["nll_calibrated"] = gaussian_nll(y_true, y_mean, cal_std)
        rep["coverage_calibrated"] = reliability_table(y_true, y_mean, cal_std)
    return rep


def format_calibration(rep: dict, title: str = "Uncertainty calibration") -> str:
    lines = [f"=== {title} ===", f"  NLL={rep['nll']:.4f}"]
    lines.append("  coverage (nominal -> empirical): " +
                 "  ".join(f"{int(k*100)}%->{v:.2f}" for k, v in rep["coverage"].items()))
    if "fitted_scale" in rep:
        lines.append(f"  fitted std-scale={rep['fitted_scale']:.3f}  NLL_cal={rep['nll_calibrated']:.4f}")
        lines.append("  coverage_cal: " +
                     "  ".join(f"{int(k*100)}%->{v:.2f}" for k, v in rep["coverage_calibrated"].items()))
    return "\n".join(lines)
