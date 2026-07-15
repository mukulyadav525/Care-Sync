"""Readers for raw Empatica E4 export files (HR, IBI, ACC, BVP, EDA, TEMP, tags)."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _read_header_series(path: Path, column: str) -> pd.Series:
    """Read a single-column E4 file: row1=start unix ts, row2=Hz, rest=values."""
    with open(path) as f:
        start_ts = float(f.readline().strip())
        sample_rate = float(f.readline().strip())
        values = np.array([float(line.strip()) for line in f if line.strip() != ""])

    n = len(values)
    index = pd.date_range(
        start=pd.Timestamp(start_ts, unit="s", tz="UTC"),
        periods=n,
        freq=pd.Timedelta(seconds=1.0 / sample_rate),
    )
    return pd.Series(values, index=index, name=column)


def _safe_read(fn, path: Path, default, label: str):
    """Call `fn(path)`; on any exception return `default` and emit a warning."""
    try:
        return fn(path)
    except FileNotFoundError:
        warnings.warn(f"[loader] {label}: file not found — {path}. Using empty data.")
        return default
    except Exception as exc:
        warnings.warn(f"[loader] {label}: could not read {path} ({exc}). Using empty data.")
        return default


def read_hr(path: Path) -> pd.Series:
    return _read_header_series(path, "HR")


def read_eda(path: Path) -> pd.Series:
    return _read_header_series(path, "EDA")


def read_temp(path: Path) -> pd.Series:
    return _read_header_series(path, "TEMP")


def read_bvp(path: Path) -> pd.Series:
    return _read_header_series(path, "BVP")


def read_acc(path: Path) -> pd.DataFrame:
    """ACC.csv: row1 = start ts (x3, identical), row2 = Hz (x3), then x,y,z in 1/64g."""
    with open(path) as f:
        start_line = f.readline().strip().split(",")
        rate_line = f.readline().strip().split(",")
        start_ts = float(start_line[0])
        sample_rate = float(rate_line[0])
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y, z = (float(v) for v in line.split(","))
            rows.append((x, y, z))

    arr = np.array(rows, dtype=float)
    # convert from 1/64 g units to g
    arr = arr / 64.0
    index = pd.date_range(
        start=pd.Timestamp(start_ts, unit="s", tz="UTC"),
        periods=len(arr),
        freq=pd.Timedelta(seconds=1.0 / sample_rate),
    )
    return pd.DataFrame(arr, index=index, columns=["ACC_x", "ACC_y", "ACC_z"])


def read_ibi(path: Path) -> pd.Series:
    """IBI.csv: row1 = "start_ts, IBI" (start ts in first field), then (offset_s, ibi_s)."""
    with open(path) as f:
        first = f.readline().strip().split(",")
        start_ts = float(first[0])
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            offset_s, ibi_s = (float(v) for v in line.split(","))
            rows.append((offset_s, ibi_s))

    if not rows:
        return pd.Series(dtype=float, name="IBI")

    arr = np.array(rows, dtype=float)
    index = pd.Timestamp(start_ts, unit="s", tz="UTC") + pd.to_timedelta(arr[:, 0], unit="s")
    return pd.Series(arr[:, 1], index=pd.DatetimeIndex(index), name="IBI")


def read_tags(path: Path) -> pd.DatetimeIndex:
    """tags_SXX.csv: one unix timestamp per line."""
    with open(path) as f:
        ts = [float(line.strip()) for line in f if line.strip()]
    return pd.DatetimeIndex(pd.to_datetime(ts, unit="s", utc=True))


def _resolve_e4_dir(subject_dir: Path) -> Path:
    """Locate the folder that actually holds the E4 CSVs.

    Handles two layouts:
      * Stress-Predict / Dataset3:  Raw_data/S01/HR.csv         (flat)
      * WESAD:                      WESAD/S2/S2_E4_Data/HR.csv   (nested)
    """
    if (subject_dir / "HR.csv").exists() or (subject_dir / "BVP.csv").exists():
        return subject_dir
    for sub in sorted(subject_dir.glob("*_E4_Data")):
        if sub.is_dir():
            return sub
    return subject_dir


def _resolve_tags(e4_dir: Path, sid: str) -> Path:
    """Tags file is `tags_S01.csv` (Stress-Predict) or `tags.csv` (WESAD)."""
    for name in (f"tags_{sid}.csv", "tags.csv"):
        if (e4_dir / name).exists():
            return e4_dir / name
    return e4_dir / f"tags_{sid}.csv"   # default; _safe_read warns if missing


def load_subject_raw(subject_dir: Path) -> dict:
    """Load all raw signals for one subject folder (e.g. Raw_data/S01).

    Missing or unreadable files are replaced with empty Series / DataFrames so
    the pipeline can continue with partial data rather than crashing. Works for
    both the flat Stress-Predict layout and WESAD's nested `<SID>_E4_Data/`.
    """
    subject_dir = Path(subject_dir)
    sid = subject_dir.name
    e4_dir = _resolve_e4_dir(subject_dir)

    _empty_series = pd.Series(dtype=float)
    _empty_acc = pd.DataFrame(columns=["ACC_x", "ACC_y", "ACC_z"])
    _empty_dti = pd.DatetimeIndex([])

    return {
        "subject_id": sid,
        "hr":   _safe_read(read_hr,   e4_dir / "HR.csv",            _empty_series, f"{sid}/HR"),
        "eda":  _safe_read(read_eda,  e4_dir / "EDA.csv",           _empty_series, f"{sid}/EDA"),
        "temp": _safe_read(read_temp, e4_dir / "TEMP.csv",          _empty_series, f"{sid}/TEMP"),
        "bvp":  _safe_read(read_bvp,  e4_dir / "BVP.csv",           _empty_series, f"{sid}/BVP"),
        "acc":  _safe_read(read_acc,  e4_dir / "ACC.csv",           _empty_acc,    f"{sid}/ACC"),
        "ibi":  _safe_read(read_ibi,  e4_dir / "IBI.csv",           _empty_series, f"{sid}/IBI"),
        "tags": _safe_read(read_tags, _resolve_tags(e4_dir, sid),   _empty_dti,    f"{sid}/tags"),
    }
