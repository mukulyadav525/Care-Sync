"""Formats the pipeline's `analysis` dict into human-readable text for the
AI report card and as the LLM prompt's "measured data" context.

Previously this just did `f"{k}: {v}"` over the raw dict, which for nested
values (hrv, stress, weekly_summary) printed literal Python dict/repr text
straight into the UI — e.g. "hrv: {'rmssd': np.float64(15.62), ...}". This
formats each known field explicitly instead.
"""

_FIELD_LABELS = {
    "rmssd": "RMSSD",
    "sdnn": "SDNN",
    "mean_rr": "Mean RR interval",
    "mean_hr": "Mean HR (from RR)",
    "pnn50": "pNN50",
    "stress_score": "Stress score",
    "stress_level": "Stress level",
}

_FIELD_UNITS = {
    "rmssd": "ms",
    "sdnn": "ms",
    "mean_rr": "ms",
    "mean_hr": "bpm",
    "pnn50": "%",
}


def _fmt_value(key: str, value) -> str:
    unit = _FIELD_UNITS.get(key, "")
    if isinstance(value, float):
        return f"{value:.2f}{(' ' + unit) if unit else ''}"
    return f"{value}{(' ' + unit) if unit else ''}"


def _format_subdict(d: dict) -> str:
    return ", ".join(f"{_FIELD_LABELS.get(k, k)}: {_fmt_value(k, v)}" for k, v in d.items())


def create_report(analysis: dict) -> str:
    lines = ["Health Summary", ""]

    if "heart_rate" in analysis:
        lines.append(f"Heart rate: {_fmt_value('mean_hr', analysis['heart_rate'])}")

    hrv = analysis.get("hrv")
    if isinstance(hrv, dict):
        lines.append(f"HRV — {_format_subdict(hrv)}")

    stress = analysis.get("stress")
    if isinstance(stress, dict):
        lines.append(f"Stress — {_format_subdict(stress)}")

    weekly = analysis.get("weekly_summary")
    if isinstance(weekly, dict):
        lines.append("Weekly trend — " + ", ".join(f"{k}: {v}" for k, v in weekly.items()))

    # Any other top-level fields we don't have a specific formatter for
    # (forward-compatible with new analysis fields) — still avoid dumping
    # raw dicts/reprs, just skip nested structures we don't recognize.
    known = {"heart_rate", "hrv", "stress", "weekly_summary", "reasoning"}
    for k, v in analysis.items():
        if k in known or isinstance(v, dict):
            continue
        lines.append(f"{k}: {v}")

    return "\n".join(lines)
