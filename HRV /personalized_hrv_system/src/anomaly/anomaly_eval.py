"""Anomaly-detection scenario evaluation (Tier-2 #16).

Given a continuous anomaly score, a boolean alert series, and a ground-truth
anomaly label, compute a standard, comparable scorecard so different scoring
rules can be tracked head-to-head across the simulator scenarios:

  threshold-free : ROC-AUC, PR-AUC (average precision)
  event-level    : event recall (caught anomaly episodes), time-to-detect
  cost-of-noise  : normal alert rate, false-alerts-per-hour, false-alert time

`detector.alert_events` is reused for run segmentation.
"""
from __future__ import annotations

import numpy as np

from .detector import alert_events


def roc_auc(score, label) -> float:
    """ROC-AUC via rank statistic (no sklearn dependency)."""
    score = np.asarray(score, float); label = np.asarray(label).astype(int)
    n_pos = int(label.sum()); n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(score, label) -> float:
    """Average precision (area under precision-recall) at all score thresholds."""
    score = np.asarray(score, float); label = np.asarray(label).astype(int)
    if label.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    lab = label[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    precision = tp / np.clip(tp + fp, 1, None)
    recall = tp / lab.sum()
    # average precision = sum over recall increments
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def event_detection(alerts, label, fs: float = 1.0) -> dict:
    """Event recall + median time-to-detect over ground-truth anomaly episodes."""
    alerts = np.asarray(alerts, bool)
    label = np.asarray(label).astype(int)
    events = alert_events(label == 1)         # ground-truth episodes
    if not events:
        return {"event_recall": float("nan"), "median_time_to_detect_s": float("nan"), "n_events": 0}
    detected, ttd = 0, []
    for a, b in events:
        fired = np.where(alerts[a:b])[0]
        if len(fired):
            detected += 1
            ttd.append(fired[0] / fs)
    return {
        "event_recall": detected / len(events),
        "median_time_to_detect_s": float(np.median(ttd)) if ttd else float("nan"),
        "n_events": len(events),
    }


def false_alert_stats(alerts, label, fs: float = 1.0) -> dict:
    """Alerts that fire OUTSIDE any ground-truth anomaly = false alarms."""
    alerts = np.asarray(alerts, bool)
    label = np.asarray(label).astype(int)
    normal = label == 0
    fa = alerts & normal
    normal_hours = normal.sum() / fs / 3600.0 if normal.sum() else 0.0
    # count contiguous false-alert runs
    fa_events = alert_events(fa)
    return {
        "normal_alert_rate": float(fa.sum() / max(normal.sum(), 1)),
        "false_alerts_per_hour": (len(fa_events) / normal_hours) if normal_hours > 0 else 0.0,
        "false_alert_time_s": float(fa.sum() / fs),
    }


def scenario_scorecard(score, alerts, label, fs: float = 1.0) -> dict:
    """Full per-scenario scorecard combining all the above."""
    return {
        "roc_auc": roc_auc(score, label),
        "pr_auc": pr_auc(score, label),
        **event_detection(alerts, label, fs),
        **false_alert_stats(alerts, label, fs),
    }


def format_scorecard(name: str, sc: dict) -> str:
    return (f"{name:<16} ROC={sc['roc_auc']:.3f} PR={sc['pr_auc']:.3f} "
            f"ev_recall={sc['event_recall']:.2f} ttd={sc['median_time_to_detect_s']:.0f}s "
            f"FA/hr={sc['false_alerts_per_hour']:.2f} normFA={sc['normal_alert_rate']:.3f}")
