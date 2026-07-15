"""Personal Digital Twin (DESIGN.md addendum: personal digital twin).

A small JSON profile capturing this person's own baseline physiology:
resting/sleep/walking/running HR, average RMSSD, and a per-hour-of-day
circadian HR table. This profile — not a population norm — is the
comparison baseline for the "digital twin score" (health-drift score):

    DigitalTwinScore(t) = (HR(t) - ExpectedHR(t)) [/ ExpectedStd(t)]

where ExpectedHR(t) is read from the circadian table for the current
hour-of-day, falling back to the state-specific baseline (resting/sleep/
walking/running) if no circadian estimate is available yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..features.circadian import circadian_profile_table
from ..features.state_classifier import EXERCISE, REST, SLEEP, WALKING

_STATE_BASELINE_KEY = {
    SLEEP: "sleep_hr",
    REST: "resting_hr",
    WALKING: "walking_hr",
    EXERCISE: "running_hr",
}


def build_digital_twin(table: pd.DataFrame, subject_id: str) -> dict:
    """Summarize a subject's full feature table into a digital twin profile.

    Requires columns: HR, physio_state, and (optionally) an RMSSD_* column.
    """
    hr = table["HR"]
    state = table["physio_state"] if "physio_state" in table.columns else pd.Series(dtype=int)

    def state_hr_mean(state_code):
        vals = hr[state == state_code] if len(state) else pd.Series(dtype=float)
        return float(vals.mean()) if len(vals) > 0 else None

    rmssd_col = next((c for c in table.columns if c.startswith("RMSSD_")), None)
    avg_rmssd = float(table[rmssd_col].mean()) if rmssd_col is not None else None

    profile = circadian_profile_table(table)
    circadian_profile = {}
    for hour, row in profile.iterrows():
        circadian_profile[str(int(hour))] = {
            "hr_mean": None if pd.isna(row["HR_mean"]) else float(row["HR_mean"]),
            "hr_std": None if pd.isna(row["HR_std"]) else float(row["HR_std"]),
        }

    return {
        "subject_id": subject_id,
        "resting_hr": state_hr_mean(REST),
        "sleep_hr": state_hr_mean(SLEEP),
        "walking_hr": state_hr_mean(WALKING),
        "running_hr": state_hr_mean(EXERCISE),
        "avg_rmssd": avg_rmssd,
        "circadian_profile": circadian_profile,
    }


def save_digital_twin(twin: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(twin, indent=2))


def load_digital_twin(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def expected_hr(twin: dict, hour_of_day: int, physio_state: int | None = None) -> tuple[float | None, float | None]:
    """Expected HR (and std, if known) for this person at `hour_of_day`,
    falling back to their state-specific baseline (sleep/rest/walk/run)."""
    entry = twin.get("circadian_profile", {}).get(str(int(hour_of_day)))
    if entry and entry.get("hr_mean") is not None:
        return entry["hr_mean"], entry.get("hr_std")

    key = _STATE_BASELINE_KEY.get(physio_state, "resting_hr")
    val = twin.get(key) or twin.get("resting_hr")
    return val, None


def digital_twin_score(current_hr: float, hour_of_day: int, twin: dict, physio_state: int | None = None) -> float:
    """Health-drift score = (current - expected) / std, or raw diff if std unknown."""
    expected, std = expected_hr(twin, hour_of_day, physio_state)
    if expected is None:
        return 0.0
    diff = current_hr - expected
    if std and std > 1e-6:
        return diff / std
    return diff
