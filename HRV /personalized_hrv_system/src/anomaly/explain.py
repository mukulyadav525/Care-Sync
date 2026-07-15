"""Explainable alerting (DESIGN.md addendum: explainable AI + illness detection).

Two pieces:

1. `illness_score` — a composite 0-1-ish score combining the three signals
   that classically co-move during infection/fever: HR elevated above
   expected, RMSSD (HRV) dropped from baseline, and TEMP elevated above the
   personal baseline. Any one signal alone may be noise; all three rising
   together is a much stronger illness signal.

2. `build_alert_reason` — turns the same numbers into a short human-readable
   string, e.g. "+HR 22bpm above expected, RMSSD dropped 35%, Temp elevated
   +1.2C, Activity: rest".
"""
from __future__ import annotations

import numpy as np


def illness_score(
    hr_zscore: np.ndarray,
    rmssd_pct_drop: np.ndarray,
    temp_deviation: np.ndarray,
) -> np.ndarray:
    """Each input should be ~0 when normal and positive when indicating illness:
      - hr_zscore: HR above expected (circadian or digital-twin z-score)
      - rmssd_pct_drop: fractional RMSSD drop from baseline (0-1+)
      - temp_deviation: degrees C above the personal TEMP baseline

    Returns the mean of the three components, each clipped at >=0 and scaled
    so that a "typical fever" (HR z~3, RMSSD drop ~35%, TEMP +1C) scores ~1.0.
    """
    hr_term = np.clip(hr_zscore, 0, None) / 3.0
    rmssd_term = np.clip(rmssd_pct_drop, 0, None) / 0.35
    temp_term = np.clip(temp_deviation, 0, None) / 1.0
    return (hr_term + rmssd_term + temp_term) / 3.0


def build_alert_reason(
    hr_actual: float,
    hr_expected: float,
    rmssd_pct_drop: float,
    temp_deviation: float,
    activity_state_name: str | None = None,
    circadian_z: float | None = None,
) -> str:
    """Compose a short human-readable reason string for an alert.

    Example: "+HR 22bpm above expected, RMSSD dropped 35%, Temp elevated +1.2C, Activity: rest"
    """
    parts: list[str] = []

    hr_diff = hr_actual - hr_expected
    if abs(hr_diff) >= 5:
        sign = "+" if hr_diff > 0 else "-"
        direction = "above" if hr_diff > 0 else "below"
        parts.append(f"{sign}HR {abs(hr_diff):.0f}bpm {direction} expected")

    if rmssd_pct_drop >= 0.15:
        parts.append(f"RMSSD dropped {rmssd_pct_drop * 100:.0f}%")

    if temp_deviation >= 0.5:
        parts.append(f"Temp elevated +{temp_deviation:.1f}C")
    elif temp_deviation <= -0.5:
        parts.append(f"Temp low {temp_deviation:.1f}C")

    if circadian_z is not None and abs(circadian_z) >= 2:
        parts.append(f"Circadian deviation z={circadian_z:.1f}")

    if activity_state_name:
        parts.append(f"Activity: {activity_state_name}")

    if not parts:
        return "No significant deviation"
    return ", ".join(parts)
