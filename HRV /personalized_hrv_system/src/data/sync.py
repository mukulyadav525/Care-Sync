"""Timestamp synchronization: align all E4 signals onto a common 1Hz grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cleaning


def _infer_fs(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    dt = (index[1] - index[0]).total_seconds()
    return 1.0 / dt if dt > 0 else 1.0


def build_synced_frame(raw: dict, cfg: dict) -> dict:
    """Combine raw per-signal series into a common-grid frame plus native-rate
    cleaned BVP/IBI for downstream feature extraction.

    Parameters
    ----------
    raw : dict from `loader.load_subject_raw`
    cfg : the loaded YAML config (uses `resample` and `cleaning` sections)

    Returns
    -------
    dict with keys:
      - "grid": pd.DataFrame indexed at `cfg.resample.freq` with columns
        HR, EDA, TEMP, ACC_mag_mean, ACC_mag_std, ACC_mag_max
      - "bvp_clean": cleaned BVP pd.Series at native rate
      - "ibi_clean": cleaned IBI pd.Series (irregular index)
      - "tags": pd.DatetimeIndex of event markers
    """
    freq = cfg["resample"]["freq"]
    clean_cfg = cfg["cleaning"]

    hr = cleaning.clip_hr(raw["hr"], clean_cfg["hr_min_bpm"], clean_cfg["hr_max_bpm"])
    eda = raw["eda"]
    temp = raw["temp"]
    acc = raw["acc"]
    bvp = raw["bvp"]
    ibi = raw["ibi"]

    # --- ACC: remove gravity, compute magnitude, aggregate to grid ---
    acc_fs = _infer_fs(acc.index)
    acc_g = cleaning.remove_gravity(acc, fs=acc_fs)
    acc_mag = np.sqrt((acc_g ** 2).sum(axis=1))
    acc_mag = pd.Series(acc_mag, index=acc.index, name="ACC_mag")
    acc_resampled = acc_mag.resample(freq).agg(["mean", "std", "max"])
    acc_resampled.columns = ["ACC_mag_mean", "ACC_mag_std", "ACC_mag_max"]

    # --- EDA: low-pass smooth then downsample (mean) ---
    eda_fs = _infer_fs(eda.index)
    eda_smooth = pd.Series(
        cleaning.lowpass_filter(eda.to_numpy(), fs=eda_fs, cutoff=clean_cfg["eda_lowpass_hz"]),
        index=eda.index,
        name="EDA",
    )
    eda_resampled = eda_smooth.resample(freq).mean()

    # --- TEMP: simple mean downsample ---
    temp_resampled = temp.resample(freq).mean().rename("TEMP")

    # --- HR: already ~1Hz, align to grid via mean (handles tiny drift) ---
    hr_resampled = hr.resample(freq).mean().rename("HR")

    # --- BVP: band-pass clean at native rate (kept native for peak detection) ---
    bvp_fs = _infer_fs(bvp.index)
    low, high = clean_cfg["bvp_bandpass_hz"]
    bvp_clean = pd.Series(
        cleaning.bandpass_filter(bvp.to_numpy(), fs=bvp_fs, low=low, high=high),
        index=bvp.index,
        name="BVP",
    )

    # --- IBI: drop implausible values, keep native (irregular) index ---
    ibi_clean = cleaning.clean_ibi(ibi, clean_cfg["ibi_min_s"], clean_cfg["ibi_max_s"])

    grid = pd.concat([hr_resampled, eda_resampled, temp_resampled, acc_resampled], axis=1)
    grid = grid.sort_index()

    # gap handling: interpolate short gaps, segment on long gaps later (build_features)
    grid = cleaning.fill_small_gaps(grid, clean_cfg["max_interpolate_gap_s"], freq=freq)

    return {
        "subject_id": raw["subject_id"],
        "grid": grid,
        "bvp_clean": bvp_clean,
        "ibi_clean": ibi_clean,
        "tags": raw["tags"],
    }
