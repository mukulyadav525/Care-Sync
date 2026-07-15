"""Physiological simulator: synthetic traces + scenario injections (DESIGN.md section 9).

Generates a 1Hz table with columns HR, ACC_mag, TEMP, EDA, RMSSD (a simplified
HRV proxy) and an `anomaly_label` column (1 during injected abnormal windows).
The same columns can be concatenated with a real subject's preprocessed grid
for "inject into real data" testing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _time_index(duration_h: float, fs_hz: float, start: str = "2026-01-01T00:00:00Z") -> pd.DatetimeIndex:
    n = int(duration_h * 3600 * fs_hz)
    return pd.date_range(start=start, periods=n, freq=pd.Timedelta(seconds=1.0 / fs_hz))


def _activity_schedule(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """A coarse daily activity-intensity schedule in [0,1], 0=sleep/rest, 1=vigorous."""
    hours = index.hour + index.minute / 60.0
    intensity = np.zeros(len(index))
    # sleep 23:00-07:00
    intensity[(hours >= 23) | (hours < 7)] = 0.0
    # light activity during the day
    day_mask = (hours >= 7) & (hours < 23)
    intensity[day_mask] = 0.15
    # a couple of exercise blocks (e.g. 08:00-08:30, 18:00-18:45)
    for start_h, end_h in [(8.0, 8.5), (18.0, 18.75)]:
        intensity[(hours >= start_h) & (hours < end_h)] = 0.9
    intensity = intensity + rng.normal(0, 0.02, size=len(index)).clip(min=0)
    return np.clip(intensity, 0, 1)


def generate_normal_day(duration_h: float = 24, fs_hz: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Scenario A: a normal day — circadian HR baseline + activity-driven response."""
    rng = np.random.default_rng(seed)
    index = _time_index(duration_h, fs_hz)
    hours = index.hour + index.minute / 60.0

    circadian = 65 + 8 * np.sin(2 * np.pi * (hours - 6) / 24.0)
    activity = _activity_schedule(index, rng)

    # first-order lag toward an activity-driven HR target
    hr_target = circadian + activity * 70.0
    hr = np.zeros(len(index))
    hr[0] = circadian[0]
    tau_up, tau_down = 90.0, 180.0
    for i in range(1, len(index)):
        tau = tau_up if hr_target[i] > hr[i - 1] else tau_down
        hr[i] = hr[i - 1] + (hr_target[i] - hr[i - 1]) / tau
    hr += rng.normal(0, 1.5, size=len(index))

    acc_mag = activity * 1.0 + rng.normal(0, 0.02, size=len(index)).clip(min=0)
    temp = 33.0 + 1.5 * np.sin(2 * np.pi * (hours - 4) / 24.0) + rng.normal(0, 0.05, size=len(index))
    eda = 0.5 + activity * 1.5 + rng.normal(0, 0.05, size=len(index))

    # RMSSD proxy: higher at rest, suppressed during activity/high HR
    rmssd = np.clip(60 - 0.4 * (hr - circadian) - 30 * activity + rng.normal(0, 3, size=len(index)), 5, 100)

    df = pd.DataFrame(
        {
            "HR": hr,
            "ACC_mag": acc_mag,
            "TEMP": temp,
            "EDA": eda,
            "RMSSD": rmssd,
            "activity_intensity": activity,
        },
        index=index,
    )
    df["anomaly_label"] = 0
    return df


def _gaussian_pulse(index: pd.DatetimeIndex, t0, amplitude: float, sigma_s: float) -> np.ndarray:
    t = (index - t0).total_seconds().to_numpy()
    return amplitude * np.exp(-(t ** 2) / (2 * sigma_s ** 2))


def inject_hr_spike(df: pd.DataFrame, onset: str, amplitude: float = 30.0, sigma_s: float = 30.0) -> pd.DataFrame:
    """Scenario B: sudden HR spike not explained by activity (e.g. panic/arrhythmia)."""
    out = df.copy()
    t0 = pd.Timestamp(onset, tz=out.index.tz)
    pulse = _gaussian_pulse(out.index, t0, amplitude, sigma_s)
    out["HR"] = out["HR"] + pulse
    out["RMSSD"] = np.clip(out["RMSSD"] - pulse * 0.8, 2, None)
    label_window = pulse > (amplitude * 0.05)
    out.loc[label_window, "anomaly_label"] = 1
    return out


def inject_gradual_drift(df: pd.DataFrame, onset: str, duration_h: float = 3.0, total_delta: float = 20.0) -> pd.DataFrame:
    """Scenario C: gradual HR drift over hours with activity unchanged (e.g. dehydration)."""
    out = df.copy()
    t0 = pd.Timestamp(onset, tz=out.index.tz)
    t1 = t0 + pd.Timedelta(hours=duration_h)
    elapsed = (out.index - t0).total_seconds() / (duration_h * 3600.0)
    ramp = np.clip(elapsed, 0, 1) * total_delta
    in_window = (out.index >= t0) & (out.index <= t1)
    out.loc[in_window, "HR"] = out.loc[in_window, "HR"] + ramp[in_window]
    out.loc[in_window, "anomaly_label"] = 1
    return out


def inject_exercise_event(df: pd.DataFrame, onset: str, duration_min: float = 30.0, target_hr: float = 150.0) -> pd.DataFrame:
    """Scenario D: exercise event — ACC rises to vigorous, HR follows with first-order lag.

    Exercise itself is treated as *normal* (not labeled anomalous) provided HR tracks
    the expected exercise response; label is left at 0 so the forecaster/anomaly
    detector is exercised on Case 2 from the brief.
    """
    out = df.copy()
    t0 = pd.Timestamp(onset, tz=out.index.tz)
    t1 = t0 + pd.Timedelta(minutes=duration_min)
    in_window = (out.index >= t0) & (out.index <= t1 + pd.Timedelta(minutes=10))  # include recovery

    hr = out["HR"].to_numpy().copy()
    acc = out["ACC_mag"].to_numpy().copy()
    idx_window = np.where(in_window)[0]
    tau_up, tau_down = 60.0, 150.0
    for i in idx_window:
        ts = out.index[i]
        exercising = t0 <= ts <= t1
        acc[i] = 0.9 if exercising else max(acc[i], 0.1)
        target = target_hr if exercising else df["HR"].iloc[i]
        tau = tau_up if target > hr[i - 1] else tau_down
        hr[i] = hr[i - 1] + (target - hr[i - 1]) / tau

    out["HR"] = hr
    out["ACC_mag"] = acc
    return out


def inject_illness_fever(df: pd.DataFrame, onset: str, duration_h: float = 12.0, temp_delta_c: float = 2.0) -> pd.DataFrame:
    """Scenario E: illness/fever — TEMP rises, resting HR rises ~2-3bpm per degC, RMSSD falls."""
    out = df.copy()
    t0 = pd.Timestamp(onset, tz=out.index.tz)
    t1 = t0 + pd.Timedelta(hours=duration_h)
    elapsed = (out.index - t0).total_seconds() / (duration_h * 3600.0)
    ramp = np.clip(elapsed, 0, 1)
    in_window = (out.index >= t0) & (out.index <= t1)

    temp_rise = ramp * temp_delta_c
    hr_rise = temp_rise * 2.5  # ~2.5 bpm per degC
    rmssd_drop = ramp * 15.0

    out.loc[in_window, "TEMP"] = out.loc[in_window, "TEMP"] + temp_rise[in_window]
    out.loc[in_window, "HR"] = out.loc[in_window, "HR"] + hr_rise[in_window]
    out.loc[in_window, "RMSSD"] = np.clip(out.loc[in_window, "RMSSD"] - rmssd_drop[in_window], 2, None)
    out.loc[in_window, "anomaly_label"] = 1
    return out


def inject_sensor_noise(
    df: pd.DataFrame, onset: str, duration_min: float = 20.0, noise_std: float = 8.0, dropout_prob: float = 0.05, seed: int = 1
) -> pd.DataFrame:
    """Scenario F: sensor noise/motion artifacts — does NOT represent a true
    physiological anomaly, so `anomaly_label` is left at 0 (tests false-alarm rate)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    t0 = pd.Timestamp(onset, tz=out.index.tz)
    t1 = t0 + pd.Timedelta(minutes=duration_min)
    in_window = (out.index >= t0) & (out.index <= t1)
    n = int(in_window.sum())

    noisy_hr = out.loc[in_window, "HR"] + rng.normal(0, noise_std, size=n)
    dropout = rng.random(n) < dropout_prob
    noisy_hr[dropout] = np.nan
    out.loc[in_window, "HR"] = noisy_hr.values

    acc_burst = out.loc[in_window, "ACC_mag"] + np.abs(rng.normal(0, 0.3, size=n))
    out.loc[in_window, "ACC_mag"] = acc_burst.values
    return out


def inject_silent_drift(
    df: pd.DataFrame, onset: str, duration_h: float = 5.0, hr_start: float = 72.0, hr_end: float = 90.0
) -> pd.DataFrame:
    """Scenario G: silent physiological drift — HR slowly rises (e.g. 72->90bpm)
    over several hours while ACC/activity stays low (resting). Unlike Scenario C
    (gradual drift on top of the existing activity pattern), here activity is
    actively suppressed to near-zero throughout, so an activity-conditioned model
    sees no explanation for the rise. RMSSD falls slightly as the drift progresses.
    Stresses the circadian-deviation and Kalman-innovation anomaly terms, which
    are designed to catch slow trends that per-step residuals can miss."""
    out = df.copy()
    t0 = pd.Timestamp(onset, tz=out.index.tz)
    t1 = t0 + pd.Timedelta(hours=duration_h)
    elapsed = (out.index - t0).total_seconds() / (duration_h * 3600.0)
    ramp = np.clip(elapsed, 0, 1)
    in_window = (out.index >= t0) & (out.index <= t1)

    hr_ramp = hr_start + (hr_end - hr_start) * ramp
    out.loc[in_window, "HR"] = hr_ramp[in_window]
    out.loc[in_window, "ACC_mag"] = np.minimum(out.loc[in_window, "ACC_mag"], 0.03)
    out.loc[in_window, "activity_intensity"] = np.minimum(out.loc[in_window, "activity_intensity"], 0.05)
    out.loc[in_window, "RMSSD"] = np.clip(out.loc[in_window, "RMSSD"] - 10 * ramp[in_window], 2, None)
    out.loc[in_window, "anomaly_label"] = 1
    return out


SCENARIOS = {
    "A_normal": lambda df: df,
    "B_hr_spike": lambda df: inject_hr_spike(df, onset=str(df.index[len(df) // 2])),
    "C_gradual_drift": lambda df: inject_gradual_drift(df, onset=str(df.index[len(df) // 3])),
    "D_exercise": lambda df: inject_exercise_event(df, onset=str(df.index[len(df) // 4])),
    "E_illness": lambda df: inject_illness_fever(df, onset=str(df.index[len(df) // 5])),
    "F_sensor_noise": lambda df: inject_sensor_noise(df, onset=str(df.index[2 * len(df) // 3])),
    "G_silent_drift": lambda df: inject_silent_drift(df, onset=str(df.index[len(df) // 6])),
}


def generate_scenario(name: str, duration_h: float = 24, fs_hz: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Generate a base normal-day trace and apply the named scenario injection."""
    base = generate_normal_day(duration_h=duration_h, fs_hz=fs_hz, seed=seed)
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Options: {list(SCENARIOS)}")
    return SCENARIOS[name](base)


def simulate_custom(
    duration_h: float = 2.0,
    fs_hz: float = 1.0,
    baseline_hr: float = 65.0,
    circadian_amplitude: float = 8.0,
    activity_level: float | None = None,
    activity_gain_bpm: float = 70.0,
    temp_baseline: float = 33.0,
    temp_offset: float = 0.0,
    temp_gain_bpm_per_degC: float = 2.5,
    eda_baseline: float = 0.5,
    eda_offset: float = 0.0,
    eda_gain_bpm: float = 10.0,
    hr_spike_amplitude: float = 0.0,
    hr_spike_time_h: float | None = None,
    hr_spike_width_s: float = 30.0,
    hr_drift_rate_bpm_per_h: float = 0.0,
    noise_std: float = 1.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a single trace where each "vital" can be dialed independently and you
    can directly observe its effect on HR.

    HR is composed as:
        HR(t) = circadian_baseline(t)
              + activity_level * activity_gain_bpm        (first-order lag)
              + (TEMP(t) - temp_baseline) * temp_gain_bpm_per_degC
              + (EDA(t) - eda_baseline) * eda_gain_bpm
              + optional Gaussian spike + linear drift
              + noise

    Parameters
    ----------
    activity_level : float in [0,1] or None
        Constant activity intensity (0=rest, 1=vigorous). If None, uses the
        default daily schedule (sleep/work/exercise blocks).
    temp_offset, eda_offset : float
        Constant shift applied to TEMP / EDA for the whole trace (e.g. simulate
        a fever by setting temp_offset=2.0).
    hr_spike_amplitude, hr_spike_time_h, hr_spike_width_s :
        Optional Gaussian HR pulse (bpm), centered at `hr_spike_time_h` hours
        into the trace (default: midpoint), width `hr_spike_width_s`.
    hr_drift_rate_bpm_per_h : float
        Linear HR drift (bpm/hour), applied across the whole trace.
    """
    rng = np.random.default_rng(seed)
    index = _time_index(duration_h, fs_hz)
    hours = index.hour + index.minute / 60.0
    t_h = np.arange(len(index)) / (3600.0 * fs_hz)  # elapsed hours from start

    circadian = baseline_hr + circadian_amplitude * np.sin(2 * np.pi * (hours - 6) / 24.0)

    if activity_level is None:
        activity = _activity_schedule(index, rng)
    else:
        activity = np.full(len(index), float(np.clip(activity_level, 0, 1)))

    temp = temp_baseline + temp_offset + 1.5 * np.sin(2 * np.pi * (hours - 4) / 24.0) + rng.normal(0, 0.05, size=len(index))
    eda = eda_baseline + eda_offset + activity * 1.5 + rng.normal(0, 0.05, size=len(index))

    hr_target = (
        circadian
        + activity * activity_gain_bpm
        + (temp - temp_baseline) * temp_gain_bpm_per_degC
        + (eda - eda_baseline) * eda_gain_bpm
        + hr_drift_rate_bpm_per_h * t_h
    )

    # first-order lag toward the target (captures realistic onset/recovery dynamics)
    hr = np.zeros(len(index))
    hr[0] = circadian[0]
    tau_up, tau_down = 90.0, 180.0
    for i in range(1, len(index)):
        tau = tau_up if hr_target[i] > hr[i - 1] else tau_down
        hr[i] = hr[i - 1] + (hr_target[i] - hr[i - 1]) / tau

    if hr_spike_amplitude:
        spike_time_h = hr_spike_time_h if hr_spike_time_h is not None else duration_h / 2.0
        t0 = pd.Timestamp(index[0]) + pd.Timedelta(hours=spike_time_h)
        hr = hr + _gaussian_pulse(index, t0, hr_spike_amplitude, hr_spike_width_s)

    hr += rng.normal(0, noise_std, size=len(index))

    acc_mag = activity * 1.0 + rng.normal(0, 0.02, size=len(index)).clip(min=0)
    rmssd = np.clip(60 - 0.4 * (hr - circadian) - 30 * activity + rng.normal(0, 3, size=len(index)), 5, 100)

    return pd.DataFrame(
        {
            "HR": hr,
            "HR_target": hr_target,
            "ACC_mag": acc_mag,
            "TEMP": temp,
            "EDA": eda,
            "RMSSD": rmssd,
            "activity_intensity": activity,
        },
        index=index,
    )
