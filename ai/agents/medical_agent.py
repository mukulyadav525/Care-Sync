"""Turns the pipeline's measured numbers into a short, data-grounded
reasoning sentence for the report/chat response.

Previously every branch here was a hardcoded string unrelated to the
actual `analysis` dict (e.g. "sleep" always claimed sleep duration was
below range, even though this pipeline never measures sleep at all — it
only has HR/HRV/stress from PPG). That's not just unhelpful, it's
actively misleading. This version only makes claims the data actually
supports, and says so plainly when it doesn't have what's being asked.
"""
from ai.models.state import AgentState


def _stress_reasoning(analysis: dict) -> str:
    stress = analysis.get("stress") or {}
    hrv = analysis.get("hrv") or {}
    level = stress.get("stress_level")
    score = stress.get("stress_score")
    rmssd = hrv.get("rmssd")
    hr = analysis.get("heart_rate")

    if level is None or score is None:
        return "Not enough data to estimate stress from this recording."

    parts = [f"Stress level is {level.lower()} (score {score}/100)."]
    if rmssd is not None and rmssd < 20:
        parts.append(f"RMSSD is low ({rmssd:.1f}ms), consistent with reduced parasympathetic recovery.")
    if hr is not None and hr > 90:
        parts.append(f"Heart rate is elevated ({hr:.0f}bpm) relative to a typical resting range.")
    if len(parts) == 1:
        parts.append("HR and HRV are both within a fairly typical range for this recording.")
    return " ".join(parts)


def _hrv_reasoning(analysis: dict) -> str:
    hrv = analysis.get("hrv") or {}
    rmssd, sdnn = hrv.get("rmssd"), hrv.get("sdnn")
    if rmssd is None:
        return "No HRV could be computed from this recording (not enough valid beats)."
    band = "low" if rmssd < 20 else "typical" if rmssd < 60 else "high"
    return (
        f"RMSSD is {rmssd:.1f}ms and SDNN is {sdnn:.1f}ms — {band} for a short recording "
        f"(rough reference bands, not a clinical threshold: <20ms low, 20-60ms typical, >60ms high variability)."
    )


def _sleep_reasoning(analysis: dict) -> str:
    # This pipeline only analyzes PPG (HR/HRV/stress) — it has no sleep
    # stage or duration data. Being explicit about that beats fabricating
    # a plausible-sounding but made-up sleep claim.
    return (
        "This analysis is based on PPG (heart rate/HRV) only — it doesn't "
        "measure sleep duration or stages. Connect a session with continuous "
        "overnight data and the digital twin's sleep-HR baseline (see HRV "
        "Insights on a session page) for a sleep-relevant signal instead."
    )


def _trend_reasoning(analysis: dict) -> str:
    weekly = analysis.get("weekly_summary")
    if not weekly:
        return "No multi-day trend data available yet — this reflects only the current recording."
    return "Trend summary: " + ", ".join(f"{k} — {v}" for k, v in weekly.items())


def _general_reasoning(analysis: dict) -> str:
    stress = analysis.get("stress") or {}
    hr = analysis.get("heart_rate")
    if not stress and hr is None:
        return "No measured data was available for this request."
    bits = []
    if hr is not None:
        bits.append(f"HR {hr:.0f}bpm")
    if stress.get("stress_level"):
        bits.append(f"stress {stress['stress_level'].lower()} ({stress.get('stress_score')}/100)")
    return f"Latest reading: {', '.join(bits)}." if bits else "Unable to determine a specific health concern from the available data."


def medical_reasoning_node(state: AgentState):
    intent = state["intent"]
    analysis = state["analysis"]

    reasoning_fns = {
        "stress": _stress_reasoning,
        "sleep": _sleep_reasoning,
        "hrv": _hrv_reasoning,
        "trend": _trend_reasoning,
    }
    reasoning = reasoning_fns.get(intent, _general_reasoning)(analysis)

    state["analysis"]["reasoning"] = reasoning

    return state
