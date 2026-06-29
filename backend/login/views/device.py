"""
Device Signal Portal API Views
===============================
Parses Empatica-E4 style session folders and returns chart-ready JSON for the
frontend "Signals" portal.

A *session* is a sub-folder inside ``Users/<username>/`` that contains the
device export files:

    ACC.csv   accelerometer (x, y, z)        -> movement magnitude (g)
    BVP.csv   raw blood-volume-pulse          -> waveform sample
    EDA.csv   electrodermal activity (uS)     -> skin conductance
    HR.csv    heart rate (bpm)
    IBI.csv   inter-beat-interval (s)         -> HRV (RMSSD / SDNN)
    TEMP.csv  skin temperature (C)
    tags*.csv event timestamps
    info.txt  free-text metadata

Empatica file layout (ACC, BVP, EDA, HR, TEMP):
    row 0 -> initial unix timestamp (UTC), one value per channel
    row 1 -> sampling rate in Hz, one value per channel
    row 2+ -> samples
IBI is different: row 0 = "<start_unix>, IBI", then rows of
"<seconds_since_start>, <ibi_seconds>".
"""
import os
import csv
import logging

import numpy as np
import pandas as pd

from django.conf import settings

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

logger = logging.getLogger(__name__)

# Signal file -> display metadata
SIGNAL_META = {
    'HR':   {'file': 'HR.csv',   'label': 'Heart Rate',          'unit': 'bpm', 'color': '#ef4444'},
    'EDA':  {'file': 'EDA.csv',  'label': 'Skin Conductance',    'unit': 'µS',  'color': '#0d9488'},
    'TEMP': {'file': 'TEMP.csv', 'label': 'Skin Temperature',    'unit': '°C',  'color': '#f59e0b'},
    'ACC':  {'file': 'ACC.csv',  'label': 'Movement (ACC mag.)', 'unit': 'g',   'color': '#10b981'},
    'BVP':  {'file': 'BVP.csv',  'label': 'Blood Volume Pulse',  'unit': '',    'color': '#6366f1'},
    'IBI':  {'file': 'IBI.csv',  'label': 'Inter-Beat Interval', 'unit': 'ms',  'color': '#ec4899'},
}

# Files that share the standard E4 header layout (single data channel we chart)
STANDARD_SIGNALS = ('HR', 'EDA', 'TEMP')

GRANULARITY_FREQ = {'minute': '1min', 'hour': '1h', 'day': '1D'}
MAX_POINTS = 2000            # cap points sent to the chart
BVP_WINDOW_SEC = 30          # raw BVP waveform sample length


# =============================================================================
# Path helpers
# =============================================================================

def _base_dir() -> str:
    return os.path.normpath(str(settings.USER_FILES_BASE_DIR))


def _session_dir(owner: str, name: str) -> str | None:
    """Resolve and sandbox a session directory path."""
    from login.security import safe_join
    path = safe_join(_base_dir(), owner, name)
    if path is None:
        return None
    return path if os.path.isdir(path) else None


def _detect_signals(session_path: str) -> list[str]:
    present = []
    for key, meta in SIGNAL_META.items():
        if os.path.isfile(os.path.join(session_path, meta['file'])):
            present.append(key)
    return present


def _has_tags(session_path: str) -> bool:
    return any(f.lower().startswith('tags') and f.lower().endswith('.csv')
               for f in os.listdir(session_path))


# =============================================================================
# Parsing
# =============================================================================

def _load_standard(path: str):
    """Load a standard single-channel E4 file -> (start_unix, fs, values)."""
    arr = np.genfromtxt(path, delimiter=',')
    if arr.ndim == 0 or arr.size < 3:
        return None
    if arr.ndim == 1:
        start, fs, values = float(arr[0]), float(arr[1]), arr[2:]
    else:  # extra columns present -> use first
        start, fs, values = float(arr[0, 0]), float(arr[1, 0]), arr[2:, 0]
    return start, fs, np.asarray(values, dtype=float)


def _load_acc(path: str):
    """Load ACC.csv -> (start_unix, fs, magnitude_in_g)."""
    arr = np.genfromtxt(path, delimiter=',')
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        return None
    start, fs = float(arr[0, 0]), float(arr[1, 0])
    xyz = arr[2:, :3].astype(float)
    mag = np.sqrt((xyz ** 2).sum(axis=1)) / 64.0  # 1/64 g units -> g
    return start, fs, mag


def _load_ibi(path: str):
    """Load IBI.csv -> (timestamps_unix, ibi_ms)."""
    with open(path, newline='') as f:
        rows = [r for r in csv.reader(f) if r]
    if len(rows) < 2:
        return None
    try:
        start = float(rows[0][0])
    except (ValueError, IndexError):
        return None
    secs, ibis = [], []
    for r in rows[1:]:
        if len(r) >= 2:
            try:
                secs.append(float(r[0]))
                ibis.append(float(r[1]))
            except ValueError:
                continue
    if not ibis:
        return None
    t_unix = start + np.asarray(secs)
    return t_unix, np.asarray(ibis) * 1000.0


def _load_tags(session_path: str) -> list[float]:
    out = []
    for fname in sorted(os.listdir(session_path)):
        if fname.lower().startswith('tags') and fname.lower().endswith('.csv'):
            with open(os.path.join(session_path, fname)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(float(line.split(',')[0]))
                    except ValueError:
                        continue
    return out


def _read_info(session_path: str) -> str | None:
    p = os.path.join(session_path, 'info.txt')
    if os.path.isfile(p):
        try:
            with open(p, encoding='utf-8', errors='replace') as f:
                return f.read()
        except OSError:
            return None
    return None


# =============================================================================
# Series / stats helpers
# =============================================================================

def _index(start: float, fs: float, n: int) -> pd.DatetimeIndex:
    return pd.to_datetime(start + np.arange(n) / fs, unit='s', utc=True)


def _downsample(s: pd.Series) -> pd.Series:
    if len(s) > MAX_POINTS:
        step = int(np.ceil(len(s) / MAX_POINTS))
        return s.iloc[::step]
    return s


def _payload(s: pd.Series) -> list[dict]:
    return [
        {'t': ts.isoformat(), 'value': round(float(v), 3)}
        for ts, v in s.items() if pd.notna(v)
    ]


def _stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if not len(a):
        return {}
    return {
        'min': round(float(a.min()), 3),
        'max': round(float(a.max()), 3),
        'avg': round(float(a.mean()), 3),
        'std': round(float(a.std()), 3),
        'count': int(a.size),
    }


def _hrv_stats(ibi_ms: np.ndarray) -> dict:
    ibi = ibi_ms[~np.isnan(ibi_ms)]
    if len(ibi) < 2:
        return {'count': int(len(ibi))}
    diff = np.diff(ibi)
    return {
        'rmssd': round(float(np.sqrt(np.mean(diff ** 2))), 2),
        'sdnn': round(float(np.std(ibi)), 2),
        'mean_ibi': round(float(np.mean(ibi)), 2),
        'mean_hr': round(float(60000.0 / np.mean(ibi)), 1),
        'count': int(len(ibi)),
    }


def _build_signal(start, fs, values, freq):
    """Resample a continuous signal to `freq` and return series + stats."""
    s = pd.Series(values, index=_index(start, fs, len(values)))
    resampled = _downsample(s.resample(freq).mean().dropna())
    return _payload(resampled), _stats(values)


def _daily_breakdown(signals: dict) -> list[dict]:
    """Per-calendar-day averages for the continuous signals."""
    frames = {}
    for key, (start, fs, values) in signals.items():
        frames[key] = pd.Series(values, index=_index(start, fs, len(values)))
    if not frames:
        return []
    df = pd.DataFrame(frames)
    daily = df.resample('1D').mean()
    out = []
    for ts, row in daily.iterrows():
        entry = {'date': ts.date().isoformat()}
        for key in frames:
            v = row.get(key)
            entry[key] = round(float(v), 2) if pd.notna(v) else None
        out.append(entry)
    return out


# =============================================================================
# Endpoints
# =============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_sessions(request):
    """
    GET /api/device/sessions/
    Lists device sessions (folders of E4 files). Superusers see every user.
    """
    base = _base_dir()
    if not os.path.isdir(base):
        return Response({'sessions': []})

    owners = (os.listdir(base) if request.user.is_superuser
              else [request.user.username])

    sessions = []
    for owner in owners:
        owner_dir = os.path.join(base, owner)
        if not os.path.isdir(owner_dir):
            continue
        for name in sorted(os.listdir(owner_dir)):
            spath = os.path.join(owner_dir, name)
            if not os.path.isdir(spath):
                continue
            present = _detect_signals(spath)
            if not present:
                continue

            start = end = None
            for key in present:
                if key == 'IBI':
                    continue
                loader = _load_acc if key == 'ACC' else _load_standard
                try:
                    res = loader(os.path.join(spath, SIGNAL_META[key]['file']))
                except Exception:
                    res = None
                if not res:
                    continue
                s_start, s_fs, vals = res
                s_end = s_start + len(vals) / s_fs if s_fs else s_start
                start = s_start if start is None else min(start, s_start)
                end = s_end if end is None else max(end, s_end)

            duration = round(end - start, 1) if (start and end) else 0
            sessions.append({
                'owner': owner,
                'name': name,
                'signals': present,
                'has_tags': _has_tags(spath),
                'start': pd.to_datetime(start, unit='s', utc=True).isoformat() if start else None,
                'end': pd.to_datetime(end, unit='s', utc=True).isoformat() if end else None,
                'duration_sec': duration,
            })

    return Response({'is_superuser': request.user.is_superuser, 'sessions': sessions})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def session_detail(request, owner, name):
    """
    GET /api/device/sessions/<owner>/<name>/?granularity=hour
    Returns parsed + aggregated data for every signal in the session.
    """
    if not request.user.is_superuser and owner != request.user.username:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)

    spath = _session_dir(owner, name)
    if spath is None:
        return Response({'error': 'Session not found.'}, status=status.HTTP_404_NOT_FOUND)

    granularity = request.GET.get('granularity', 'hour')
    freq = GRANULARITY_FREQ.get(granularity, '1h')

    present = _detect_signals(spath)
    if not present:
        return Response({'error': 'No recognised device files in this session.'},
                        status=status.HTTP_400_BAD_REQUEST)

    signals_out = {}
    continuous = {}   # key -> (start, fs, values) for daily breakdown
    overall_start = overall_end = None

    for key in present:
        meta = SIGNAL_META[key]
        fpath = os.path.join(spath, meta['file'])
        try:
            if key == 'IBI':
                res = _load_ibi(fpath)
                if not res:
                    continue
                t_unix, ibi_ms = res
                s = _downsample(pd.Series(ibi_ms, index=pd.to_datetime(t_unix, unit='s', utc=True)))
                signals_out[key] = {
                    **_meta_public(key),
                    'mode': 'event',
                    'series': _payload(s),
                    'stats': _hrv_stats(ibi_ms),
                }
                continue

            if key == 'BVP':
                res = _load_standard(fpath)
                if not res:
                    continue
                start, fs, values = res
                n = min(len(values), int(fs * BVP_WINDOW_SEC))
                seg = pd.Series(values[:n], index=_index(start, fs, n))
                signals_out[key] = {
                    **_meta_public(key),
                    'mode': 'waveform',
                    'sample_rate': fs,
                    'window_sec': BVP_WINDOW_SEC,
                    'series': _payload(_downsample(seg)),
                    'stats': _stats(values),
                }
                _track = (start, fs, len(values))
                overall_start, overall_end = _extend(overall_start, overall_end, *_track)
                continue

            # Continuous signals: HR / EDA / TEMP / ACC
            res = _load_acc(fpath) if key == 'ACC' else _load_standard(fpath)
            if not res:
                continue
            start, fs, values = res
            series, stats = _build_signal(start, fs, values, freq)
            signals_out[key] = {
                **_meta_public(key),
                'mode': 'continuous',
                'sample_rate': fs,
                'series': series,
                'stats': stats,
            }
            continuous[key] = (start, fs, values)
            overall_start, overall_end = _extend(overall_start, overall_end, start, fs, len(values))

        except Exception as e:  # noqa: BLE001 - never let one bad file break the page
            logger.error("device session_detail %s/%s [%s]: %s", owner, name, key, e)
            continue

    tags = _load_tags(spath)

    # Evaluate alert rules against mean values of this session
    signal_means = {
        k: {'mean': v['stats'].get('avg')}
        for k, v in signals_out.items()
        if v.get('mode') == 'continuous' and v.get('stats')
    }
    from login.views.alerts import evaluate_alerts, persist_fired_alerts
    fired_alerts = evaluate_alerts(request.user, signal_means)
    if fired_alerts:
        persist_fired_alerts(request.user, fired_alerts, owner, name)

    return Response({
        'owner': owner,
        'name': name,
        'granularity': granularity,
        'start': pd.to_datetime(overall_start, unit='s', utc=True).isoformat() if overall_start else None,
        'end': pd.to_datetime(overall_end, unit='s', utc=True).isoformat() if overall_end else None,
        'duration_sec': round(overall_end - overall_start, 1) if (overall_start and overall_end) else 0,
        'info': _read_info(spath),
        'tags': [pd.to_datetime(t, unit='s', utc=True).isoformat() for t in tags],
        'signals': signals_out,
        'daily': _daily_breakdown(continuous),
        'alerts': fired_alerts,
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def session_trends(request):
    """
    GET /api/device/sessions/trends/?owner=<username>
    Returns per-session averages for all continuous signals so the frontend
    can plot how metrics evolve over time across sessions.
    """
    target = request.GET.get('owner', request.user.username)
    if not request.user.is_superuser and target != request.user.username:
        return Response({'error': 'Access denied.'}, status=status.HTTP_403_FORBIDDEN)

    base = _base_dir()
    owner_dir = os.path.join(base, target)
    if not os.path.isdir(owner_dir):
        return Response({'trends': []})

    results = []
    for name in sorted(os.listdir(owner_dir)):
        spath = os.path.join(owner_dir, name)
        if not os.path.isdir(spath):
            continue
        present = _detect_signals(spath)
        if not present:
            continue

        session_start = None
        avgs: dict = {}

        for key in ('HR', 'EDA', 'TEMP', 'ACC'):
            if key not in present:
                continue
            fpath = os.path.join(spath, SIGNAL_META[key]['file'])
            try:
                res = _load_acc(fpath) if key == 'ACC' else _load_standard(fpath)
                if not res:
                    continue
                s_start, _, values = res
                vals = np.asarray(values, dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    avgs[key] = round(float(np.mean(vals)), 2)
                    if session_start is None:
                        session_start = s_start
            except Exception:
                continue

        # HRV from IBI
        if 'IBI' in present:
            ibi_path = os.path.join(spath, SIGNAL_META['IBI']['file'])
            try:
                r = _load_ibi(ibi_path)
                if r is not None:
                    _, ibi_ms = r
                    ibi_ms = ibi_ms[np.isfinite(ibi_ms)]
                    if len(ibi_ms) >= 2:
                        diffs = np.diff(ibi_ms)
                        avgs['RMSSD'] = round(float(np.sqrt(np.mean(diffs ** 2))), 2)
            except Exception:
                pass

        if not avgs:
            continue

        results.append({
            'name': name,
            'start': pd.to_datetime(session_start, unit='s', utc=True).isoformat() if session_start else None,
            **avgs,
        })

    return Response({'owner': target, 'trends': results})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def compare_sessions(request):
    """
    GET /api/device/sessions/compare/?a=owner/name&b=owner/name&granularity=hour
    Returns two sessions' stats + daily tables for side-by-side comparison.
    """
    a_param = request.GET.get('a', '')
    b_param = request.GET.get('b', '')
    granularity = request.GET.get('granularity', 'hour')
    freq = GRANULARITY_FREQ.get(granularity, '1h')

    def _parse_session(param):
        parts = param.strip('/').split('/', 1)
        if len(parts) != 2:
            return None, None
        return parts[0], parts[1]

    def _load_one(owner, name):
        if not request.user.is_superuser and owner != request.user.username:
            return None, 'Access denied.'
        spath = _session_dir(owner, name)
        if spath is None:
            return None, f'Session {owner}/{name} not found.'
        present = _detect_signals(spath)
        stats = {}
        for key in present:
            if key in ('BVP', 'IBI'):
                continue
            fpath = os.path.join(spath, SIGNAL_META[key]['file'])
            try:
                res = _load_acc(fpath) if key == 'ACC' else _load_standard(fpath)
                if not res:
                    continue
                _, _, values = res
                stats[key] = _stats(values)
                stats[key]['label'] = SIGNAL_META[key]['label']
                stats[key]['unit'] = SIGNAL_META[key]['unit']
                stats[key]['color'] = SIGNAL_META[key]['color']
            except Exception:
                continue
        return {'owner': owner, 'name': name, 'stats': stats}, None

    owner_a, name_a = _parse_session(a_param)
    owner_b, name_b = _parse_session(b_param)
    if not (owner_a and owner_b):
        return Response({'error': 'Provide ?a=owner/session&b=owner/session'}, status=400)

    result_a, err_a = _load_one(owner_a, name_a)
    result_b, err_b = _load_one(owner_b, name_b)

    errors = [e for e in (err_a, err_b) if e]
    if errors:
        return Response({'error': ' | '.join(errors)}, status=400)

    return Response({'a': result_a, 'b': result_b})


# --- small internal helpers -------------------------------------------------

def _meta_public(key: str) -> dict:
    m = SIGNAL_META[key]
    return {'key': key, 'label': m['label'], 'unit': m['unit'], 'color': m['color']}


def _extend(cur_start, cur_end, start, fs, n):
    end = start + (n / fs if fs else 0)
    new_start = start if cur_start is None else min(cur_start, start)
    new_end = end if cur_end is None else max(cur_end, end)
    return new_start, new_end
