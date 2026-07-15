"""Signal cleaning: filtering, artifact removal, and gap handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sps


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """Butterworth band-pass filter, used to clean BVP."""
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    b, a = sps.butter(order, [low_n, high_n], btype="band")
    return sps.filtfilt(b, a, x)


def lowpass_filter(x: np.ndarray, fs: float, cutoff: float, order: int = 3) -> np.ndarray:
    """Butterworth low-pass filter, used to clean EDA / TEMP."""
    nyq = fs / 2.0
    cutoff_n = min(cutoff / nyq, 0.999)
    b, a = sps.butter(order, cutoff_n, btype="low")
    return sps.filtfilt(b, a, x)


def remove_gravity(acc_df: pd.DataFrame, fs: float, cutoff: float = 0.5) -> pd.DataFrame:
    """High-pass filter each ACC axis to remove the gravity (DC) component."""
    nyq = fs / 2.0
    cutoff_n = min(cutoff / nyq, 0.999)
    b, a = sps.butter(3, cutoff_n, btype="high")
    out = acc_df.copy()
    for col in acc_df.columns:
        out[col] = sps.filtfilt(b, a, acc_df[col].to_numpy())
    return out


def clip_hr(hr: pd.Series, hr_min: float, hr_max: float) -> pd.Series:
    return hr.clip(lower=hr_min, upper=hr_max)


def clean_ibi(ibi: pd.Series, ibi_min: float, ibi_max: float) -> pd.Series:
    """Drop physiologically implausible inter-beat intervals."""
    return ibi[(ibi >= ibi_min) & (ibi <= ibi_max)]


def fill_small_gaps(df: pd.DataFrame, max_gap_s: int, freq: str = "1s") -> pd.DataFrame:
    """Linearly interpolate gaps <= max_gap_s; leave longer gaps as NaN.

    Adds a `<col>_stale` flag (1 if the value at this timestep was interpolated
    across a gap > 1 sample, 0 otherwise) for columns that had any NaNs.
    """
    out = df.copy()
    max_gap_samples = int(pd.Timedelta(max_gap_s, unit="s") / pd.Timedelta(freq))
    for col in df.columns:
        s = df[col]
        na_mask = s.isna()
        if not na_mask.any():
            continue
        interpolated = s.interpolate(method="linear", limit=max_gap_samples, limit_area="inside")
        out[col] = interpolated
        out[f"{col}_stale"] = na_mask.astype(int) & interpolated.notna().astype(int)
    return out


def segment_by_gaps(df: pd.DataFrame, max_segment_gap_s: int, freq: str = "1s") -> list[pd.DataFrame]:
    """Split a DataFrame into contiguous segments wherever a column has a gap
    longer than max_segment_gap_s (after small-gap interpolation)."""
    if df.empty:
        return []
    still_missing = df.isna().any(axis=1)
    # group consecutive valid rows
    segments = []
    start = None
    for i, (ts, missing) in enumerate(still_missing.items()):
        if not missing and start is None:
            start = ts
        elif missing and start is not None:
            segments.append(df.loc[start: df.index[i - 1]])
            start = None
    if start is not None:
        segments.append(df.loc[start:])
    # drop trivially short segments
    min_len = max(1, int(pd.Timedelta(max_segment_gap_s, unit="s") / pd.Timedelta(freq)))
    return [seg for seg in segments if len(seg) >= min_len]
