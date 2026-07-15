"""Computes a genuine weekly trend from the same sample dataset the /trends
route uses, instead of the previous hardcoded "Higher than last week" /
"Slightly reduced" strings — those were fabricated and shown regardless of
what the data actually did, which is worse than not answering.

This is still a simplification (compares first half vs second half of the
one demo CSV, not real multi-week per-user history — there's no persisted
per-user analysis history yet for that), but every number in it is real.
"""
from pathlib import Path

import pandas as pd

from ai.models.state import AgentState

_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_ppg.csv"


def _compute_trend() -> dict | None:
    try:
        df = pd.read_csv(_CSV_PATH)
    except Exception:
        return None
    if df.empty or "stress_score" not in df.columns:
        return None

    mid = len(df) // 2
    first_half, second_half = df.iloc[:mid], df.iloc[mid:]
    if first_half.empty or second_half.empty:
        return None

    stress_delta = float(second_half["stress_score"].mean() - first_half["stress_score"].mean())
    rmssd_delta = (
        float(second_half["rmssd"].mean() - first_half["rmssd"].mean())
        if "rmssd" in df.columns else None
    )

    def _direction(delta: float, higher_is_worse: bool) -> str:
        if abs(delta) < 1:
            return "roughly stable"
        worse = delta > 0 if higher_is_worse else delta < 0
        return "trending up" if (delta > 0) else "trending down"

    summary = {"stress": f"{_direction(stress_delta, True)} ({stress_delta:+.1f} pts)"}
    if rmssd_delta is not None:
        summary["hrv_rmssd"] = f"{_direction(rmssd_delta, False)} ({rmssd_delta:+.1f}ms)"
    return summary


def trend_node(state: AgentState):
    if state["intent"] != "trend":
        return state

    summary = _compute_trend()
    if summary:
        state["analysis"]["weekly_summary"] = summary
    # If we couldn't compute anything real, deliberately leave
    # weekly_summary unset — medical_agent's trend reasoning already
    # handles that case honestly ("No multi-day trend data available yet").

    return state
