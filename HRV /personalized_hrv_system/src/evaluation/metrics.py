"""Comprehensive metric suite for HR / HRV forecasting.

Two families:
  * regression_metrics / forecast_report — for HR (and RMSSD) forecasting: MAE,
    RMSE, MAPE, R^2, bias / mean-deviation, std of error, median-AE, max-AE,
    Pearson r, and prediction-interval coverage.
  * classification_metrics — generic classifier scoring (kept for any categorical
    sub-task): accuracy, balanced accuracy, per-class & averaged precision/recall/
    F1, specificity, dice (= F1), Jaccard/IoU, full confusion matrix, raw
    TP / TN / FP / FN per class, plus Cohen's kappa and MCC.

Pure NumPy so it can be called from any pipeline without pulling in torch.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ regression


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizon_names: list[str] | None = None) -> dict:
    """Per-horizon and overall regression metrics.

    `y_true`, `y_pred` are (N, H) arrays (H = number of horizons / targets).
    Returns a nested dict: {"overall": {...}, "per_horizon": {name: {...}}}.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]

    n_h = y_true.shape[1]
    if horizon_names is None:
        horizon_names = [f"h{j}" for j in range(n_h)]

    per_horizon = {}
    for j in range(n_h):
        per_horizon[horizon_names[j]] = _regression_single(y_true[:, j], y_pred[:, j])

    overall = _regression_single(y_true.ravel(), y_pred.ravel())
    return {"overall": overall, "per_horizon": per_horizon}


def _regression_single(yt: np.ndarray, yp: np.ndarray) -> dict:
    err = yp - yt
    abs_err = np.abs(err)
    denom = np.clip(np.abs(yt), 1e-6, None)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) if len(yt) else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if len(yt) > 1 and yt.std() > 0 and yp.std() > 0:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson = float("nan")

    return {
        "MAE": float(abs_err.mean()) if len(abs_err) else float("nan"),
        "RMSE": float(np.sqrt((err ** 2).mean())) if len(err) else float("nan"),
        "MAPE": float((abs_err / denom).mean() * 100) if len(err) else float("nan"),
        "R2": r2,
        "bias": float(err.mean()) if len(err) else float("nan"),       # mean deviation (signed)
        "mean_deviation": float(abs_err.mean()) if len(err) else float("nan"),
        "std_error": float(err.std()) if len(err) else float("nan"),
        "median_AE": float(np.median(abs_err)) if len(abs_err) else float("nan"),
        "max_AE": float(abs_err.max()) if len(abs_err) else float("nan"),
        "pearson_r": pearson,
        "n": int(len(yt)),
    }


# -------------------------------------------------------------- classification


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> np.ndarray:
    """Rows = true class, cols = predicted class, ordered as `labels`."""
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def classification_metrics(y_true, y_pred, labels: list | None = None) -> dict:
    """Full classification report with raw TP/TN/FP/FN per class.

    Works for binary and multi-class. For each class c (one-vs-rest):
      TP = predicted c and is c
      FP = predicted c but is not c
      FN = is c but predicted other
      TN = neither predicted nor is c
    Dice == F1 for the one-vs-rest binarisation; IoU == Jaccard.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    cm = confusion_matrix(y_true, y_pred, labels)
    total = cm.sum()
    correct = int(np.trace(cm))

    per_class = {}
    recalls, precisions, f1s, supports = [], [], [], []
    for i, lab in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0          # sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        dice = f1                                                    # identical for binary OVR
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0     # Jaccard

        per_class[str(lab)] = {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "sensitivity": recall,
            "specificity": specificity, "f1": f1, "dice": dice, "iou": iou,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    supports = np.array(supports, dtype=float)
    w = supports / supports.sum() if supports.sum() > 0 else np.ones_like(supports) / len(supports)

    accuracy = correct / total if total > 0 else 0.0
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0

    macro = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "dice": float(np.mean(f1s)) if f1s else 0.0,
    }
    weighted = {
        "precision": float(np.sum(np.array(precisions) * w)),
        "recall": float(np.sum(np.array(recalls) * w)),
        "f1": float(np.sum(np.array(f1s) * w)),
    }

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": _cohen_kappa(cm),
        "mcc": _mcc(cm),
        "macro_avg": macro,
        "weighted_avg": weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": [str(lab) for lab in labels],
        "n": int(total),
    }


def _cohen_kappa(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return float("nan")
    po = np.trace(cm) / total
    row = cm.sum(axis=1) / total
    col = cm.sum(axis=0) / total
    pe = float(np.sum(row * col))
    return float((po - pe) / (1 - pe)) if (1 - pe) > 0 else float("nan")


def _mcc(cm: np.ndarray) -> float:
    """Multi-class Matthews correlation coefficient (Gorodkin generalisation)."""
    t_k = cm.sum(axis=1).astype(float)   # true totals per class
    p_k = cm.sum(axis=0).astype(float)   # predicted totals per class
    c = float(np.trace(cm))
    s = float(cm.sum())
    if s == 0:
        return float("nan")
    cov_ytyp = c * s - float(np.dot(t_k, p_k))
    cov_ypyp = s * s - float(np.dot(p_k, p_k))
    cov_ytyt = s * s - float(np.dot(t_k, t_k))
    denom = np.sqrt(cov_ypyp * cov_ytyt)
    return float(cov_ytyp / denom) if denom > 0 else float("nan")


# ----------------------------------------------------------------- formatting


def _parse_target(name: str) -> tuple[str, str]:
    """'HR_target_60s' -> ('HR', '60s'); 'RMSSD_target_300s' -> ('RMSSD','300s')."""
    if "_target_" in name:
        grp, hor = name.split("_target_", 1)
        return grp, hor
    return name, ""


def prediction_interval_coverage(y_true, y_mean, y_std, levels=(0.90, 0.95)) -> dict:
    """Fraction of true values inside the predicted Gaussian interval at each level.

    Well-calibrated uncertainty => coverage ~= the nominal level (0.90, 0.95).
    Returns {"cov@0.90": .., "cov@0.95": .., "mean_std": ..}.
    """
    y_true = np.asarray(y_true, float); y_mean = np.asarray(y_mean, float)
    y_std = np.clip(np.asarray(y_std, float), 1e-6, None)
    z = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    out = {}
    for lv in levels:
        c = z.get(lv, 1.96)
        inside = np.abs(y_true - y_mean) <= c * y_std
        out[f"cov@{lv:.2f}"] = float(inside.mean()) if len(y_true) else float("nan")
    out["mean_std"] = float(y_std.mean()) if len(y_std) else float("nan")
    return out


def forecast_report(y_true, y_pred, y_std, target_cols: list[str]) -> dict:
    """Per-target forecasting report, grouped by signal (HR vs RMSSD) and horizon,
    each with regression metrics + prediction-interval coverage.

    y_true, y_pred, y_std are (N, T) with columns ordered like `target_cols`.
    Returns {"groups": {"HR": {"60s": {...}, ...}, "RMSSD": {...}}, "overall": {...}}.
    """
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    y_std = None if y_std is None else np.asarray(y_std, float)

    groups: dict[str, dict] = {}
    for j, name in enumerate(target_cols):
        grp, hor = _parse_target(name)
        m = _regression_single(y_true[:, j], y_pred[:, j])
        if y_std is not None:
            m.update(prediction_interval_coverage(y_true[:, j], y_pred[:, j], y_std[:, j]))
        groups.setdefault(grp, {})[hor] = m

    overall = _regression_single(y_true.ravel(), y_pred.ravel())
    return {"groups": groups, "overall": overall}


def per_subject_metrics(subject_results: list[tuple], target_cols: list[str]) -> dict:
    """Aggregate per-subject regression metrics into a distribution.

    `subject_results` = list of (label, y_true(N,T), y_pred(N,T)).
    Returns {"per_subject": {label: {MAE,RMSE,R2,...}}, "summary": {metric: {mean,std,min,max}}}.
    """
    per = {}
    for label, yt, yp in subject_results:
        per[label] = _regression_single(np.asarray(yt, float).ravel(), np.asarray(yp, float).ravel())
    summary = {}
    for key in ("MAE", "RMSE", "MAPE", "R2", "bias", "pearson_r"):
        vals = np.array([m[key] for m in per.values() if np.isfinite(m[key])], float)
        if len(vals):
            summary[key] = {"mean": float(vals.mean()), "std": float(vals.std()),
                            "min": float(vals.min()), "max": float(vals.max())}
    return {"per_subject": per, "summary": summary}


def format_forecast_report(report: dict, title: str = "Forecast report") -> str:
    lines = [f"=== {title} ==="]
    for grp, horizons in report["groups"].items():
        lines.append(f"  [{grp}]")
        for hor, m in horizons.items():
            cov = ""
            if "cov@0.95" in m:
                cov = f"  cov90={m['cov@0.90']:.2f} cov95={m['cov@0.95']:.2f}"
            lines.append(
                f"    {hor:>6}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
                f"R2={m['R2']:.3f}  r={m['pearson_r']:.3f}  bias={m['bias']:+.3f}{cov}"
            )
    return "\n".join(lines)


def format_per_subject(report: dict, title: str = "Per-subject metrics") -> str:
    lines = [f"=== {title} ==="]
    for metric, s in report["summary"].items():
        lines.append(f"  {metric:>9}: mean={s['mean']:.3f}  std={s['std']:.3f}  "
                     f"min={s['min']:.3f}  max={s['max']:.3f}")
    return "\n".join(lines)


def format_regression(metrics: dict, title: str = "Regression metrics") -> str:
    lines = [f"=== {title} ==="]
    o = metrics["overall"]
    lines.append(
        f"  OVERALL  MAE={o['MAE']:.3f}  RMSE={o['RMSE']:.3f}  MAPE={o['MAPE']:.2f}%  "
        f"R2={o['R2']:.3f}  bias={o['bias']:+.3f}  r={o['pearson_r']:.3f}  n={o['n']}"
    )
    for name, m in metrics["per_horizon"].items():
        lines.append(
            f"   [{name:>10}] MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
            f"MAPE={m['MAPE']:.2f}%  R2={m['R2']:.3f}  bias={m['bias']:+.3f}"
        )
    return "\n".join(lines)


def format_classification(metrics: dict, title: str = "Classification metrics") -> str:
    lines = [f"=== {title} ==="]
    lines.append(
        f"  accuracy={metrics['accuracy']:.4f}  balanced_acc={metrics['balanced_accuracy']:.4f}  "
        f"kappa={metrics['cohen_kappa']:.4f}  mcc={metrics['mcc']:.4f}  n={metrics['n']}"
    )
    ma, wa = metrics["macro_avg"], metrics["weighted_avg"]
    lines.append(f"  macro    P={ma['precision']:.3f} R={ma['recall']:.3f} F1={ma['f1']:.3f} dice={ma['dice']:.3f}")
    lines.append(f"  weighted P={wa['precision']:.3f} R={wa['recall']:.3f} F1={wa['f1']:.3f}")
    lines.append(f"  {'class':>10} {'TP':>7} {'TN':>7} {'FP':>7} {'FN':>7} {'prec':>6} {'rec':>6} {'spec':>6} {'F1':>6} {'dice':>6} {'IoU':>6} {'supp':>7}")
    for lab, c in metrics["per_class"].items():
        lines.append(
            f"  {lab:>10} {c['TP']:>7} {c['TN']:>7} {c['FP']:>7} {c['FN']:>7} "
            f"{c['precision']:>6.3f} {c['recall']:>6.3f} {c['specificity']:>6.3f} "
            f"{c['f1']:>6.3f} {c['dice']:>6.3f} {c['iou']:>6.3f} {c['support']:>7}"
        )
    return "\n".join(lines)
