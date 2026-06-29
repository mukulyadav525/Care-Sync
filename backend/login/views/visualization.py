"""
Visualization API Views
Handles: PPG visualization, GSR visualization, Actigraphy data
All views return JSON data for the Next.js frontend to render charts.
"""
import os
import io
import csv
import logging
import chardet
from datetime import datetime, time, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import get_object_or_404

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from login.models import GoogleSheet

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _get_user_file_path(username: str, filename: str) -> str | None:
    """Return the absolute path to a user file, or None if not safe."""
    from login.security import safe_join
    return safe_join(str(settings.USER_FILES_BASE_DIR), username, filename)


def _read_csv_with_encoding(file_path: str) -> pd.DataFrame:
    """Read a CSV file, automatically detecting its encoding."""
    with open(file_path, 'rb') as f:
        raw = f.read()
    enc = chardet.detect(raw).get('encoding', 'utf-8') or 'utf-8'
    return pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines='skip')


def ppg_time_to_seconds(time_str: str) -> float | None:
    """Convert PPG time string (HH:MM:SS or HH:MM:SS.sss) to total seconds."""
    try:
        if '.' in time_str:
            time_part, ms_part = time_str.split('.')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part.ljust(3, '0')[:3])
            return h * 3600 + m * 60 + s + ms / 1000
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
    except (ValueError, IndexError, AttributeError):
        return None


def seconds_to_time_str(seconds: float) -> str | None:
    """Convert seconds back to HH:MM:SS.sss string."""
    try:
        if seconds is None or seconds < 0:
            return None
        total_sec = int(seconds)
        ms = int((seconds % 1) * 1000)
        h = (total_sec // 3600) % 24
        m = (total_sec % 3600) // 60
        s = total_sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    except Exception:
        return None


def calculate_ppg_stats(ppg_array: np.ndarray, sampling_rate: int = 100) -> dict:
    """Calculate heart rate and HRV stats from a PPG array."""
    try:
        peaks, _ = find_peaks(ppg_array, height=np.mean(ppg_array), distance=50)
        base = {
            'min': round(float(np.min(ppg_array)), 2),
            'max': round(float(np.max(ppg_array)), 2),
            'avg': round(float(np.mean(ppg_array)), 2),
            'median': round(float(np.median(ppg_array)), 2),
        }
        if len(peaks) < 2:
            base['hr'] = None
            base['hrv'] = None
            return base
        intervals = np.diff(peaks)
        heart_rates = 60 / (intervals / sampling_rate)
        rr_ms = intervals * (1000 / sampling_rate)
        base['hr'] = round(float(np.mean(heart_rates)), 2)
        base['hrv'] = round(float(np.sqrt(np.mean(np.square(np.diff(rr_ms))))), 2)
        return base
    except Exception as e:
        logger.error(f"PPG stats error: {e}")
        return {}


def process_csv_into_days(rows: list, signal_col: str, time_col: str = 'Time',
                           day_filter: int = None, from_sec: float = None, to_sec: float = None):
    """
    Split CSV rows into days based on timestamp resets.
    Returns: (daily_values, daily_times) lists of lists.
    """
    daily_values = []
    daily_times = []
    cur_vals, cur_times = [], []
    prev_sec = None
    cur_day = 1

    for row in rows:
        if signal_col not in row or time_col not in row:
            continue
        t_sec = ppg_time_to_seconds(str(row[time_col]))
        if t_sec is None:
            continue
        try:
            val = float(row[signal_col])
        except (ValueError, TypeError):
            continue

        # Detect day boundary
        if prev_sec is not None and t_sec < prev_sec - 10:
            if day_filter is not None and cur_day == day_filter:
                break
            if cur_vals:
                daily_values.append(cur_vals)
                daily_times.append(cur_times)
            cur_vals, cur_times = [], []
            cur_day += 1

        # Apply filters
        if day_filter is not None and cur_day != day_filter:
            prev_sec = t_sec
            continue
        if from_sec is not None and to_sec is not None:
            if not (from_sec <= t_sec <= to_sec):
                prev_sec = t_sec
                continue

        t_str = seconds_to_time_str(t_sec)
        if t_str:
            cur_vals.append(val)
            cur_times.append(t_str)
            prev_sec = t_sec

    if cur_vals:
        daily_values.append(cur_vals)
        daily_times.append(cur_times)

    return daily_values, daily_times


def prepare_chart_series(times: list, values: list, label: str = 'value') -> list:
    """
    Convert time strings and values into chart-ready {time, value} dicts.
    Times are ISO-ish strings; chart libraries like Chart.js / Recharts handle them.
    """
    return [{'time': t, label: round(v, 4)} for t, v in zip(times, values)]


def normalize_csv_to_ppg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either pre-computed PPG/Time columns, or raw device columns
    (Hour/Minute/Second/Millisecond/Red/IR), and return a normalized DataFrame.
    """
    if {'Time', 'PPG'}.issubset(df.columns):
        return df[['Time', 'PPG']].dropna()

    required = {'Hour', 'Minute', 'Second', 'Millisecond', 'Red', 'IR'}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df['Time'] = df.apply(
        lambda r: f"{int(r['Hour']):02d}:{int(r['Minute']):02d}:{int(r['Second']):02d}.{int(r['Millisecond']):03d}",
        axis=1
    )
    df['PPG'] = df.apply(lambda r: r['Red'] / r['IR'] if r['IR'] != 0 else None, axis=1)
    return df[['Time', 'PPG']].dropna()


# =============================================================================
# PPG VIEWS
# =============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def view_local_ppg(request, filename):
    """
    GET /api/visualization/local-ppg/<filename>/
    Query params: ?day=1&from=HH:MM&to=HH:MM
    Returns multi-day PPG chart data as JSON.
    """
    username = request.user.username
    file_path = _get_user_file_path(username, filename)
    if file_path is None:
        return Response({'error': 'Access denied.'}, status=status.HTTP_403_FORBIDDEN)
    if not os.path.exists(file_path):
        return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

    try:
        df = _read_csv_with_encoding(file_path)
        df = normalize_csv_to_ppg(df)
        if df.empty:
            return Response({'error': 'CSV missing required columns (Time+PPG or Hour/Minute/Second/Millisecond/Red/IR).'}, status=400)

        rows = df.to_dict(orient='records')

        # Parse filter params
        day_filter = request.GET.get('day')
        from_time = request.GET.get('from')
        to_time = request.GET.get('to')

        day_int = int(day_filter) if day_filter else None
        from_sec = ppg_time_to_seconds(from_time + ':00') if from_time else None
        to_sec = ppg_time_to_seconds(to_time + ':00') if to_time else None

        daily_vals, daily_times = process_csv_into_days(rows, 'PPG', day_filter=day_int, from_sec=from_sec, to_sec=to_sec)

        days_data = []
        for i, (vals, times) in enumerate(zip(daily_vals, daily_times), start=1):
            arr = np.array(vals)
            days_data.append({
                'day': i,
                'series': prepare_chart_series(times, vals, 'ppg'),
                'stats': calculate_ppg_stats(arr),
            })

        return Response({
            'filename': filename,
            'username': username,
            'total_days': len(days_data),
            'days': days_data,
        })

    except Exception as e:
        logger.error(f"view_local_ppg error: {e}")
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def view_local_gsr(request, filename):
    """
    GET /api/visualization/local-gsr/<filename>/
    Query params: ?day=1&from=HH:MM&to=HH:MM
    Returns multi-day GSR chart data as JSON.
    """
    username = request.user.username
    file_path = _get_user_file_path(username, filename)
    if file_path is None:
        return Response({'error': 'Access denied.'}, status=status.HTTP_403_FORBIDDEN)
    if not os.path.exists(file_path):
        return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

    try:
        df = _read_csv_with_encoding(file_path)

        # Normalize: handle raw device format
        if 'GSR' not in df.columns:
            return Response({'error': 'CSV does not contain a GSR column.'}, status=400)

        if 'Time' not in df.columns and {'Hour', 'Minute', 'Second', 'Millisecond'}.issubset(df.columns):
            df['Time'] = df.apply(
                lambda r: f"{int(r['Hour']):02d}:{int(r['Minute']):02d}:{int(r['Second']):02d}.{int(r['Millisecond']):03d}",
                axis=1
            )

        df = df[['Time', 'GSR']].dropna()
        rows = df.to_dict(orient='records')

        day_filter = request.GET.get('day')
        from_time = request.GET.get('from')
        to_time = request.GET.get('to')

        day_int = int(day_filter) if day_filter else None
        from_sec = ppg_time_to_seconds(from_time + ':00') if from_time else None
        to_sec = ppg_time_to_seconds(to_time + ':00') if to_time else None

        daily_vals, daily_times = process_csv_into_days(rows, 'GSR', day_filter=day_int, from_sec=from_sec, to_sec=to_sec)

        days_data = []
        for i, (vals, times) in enumerate(zip(daily_vals, daily_times), start=1):
            arr = np.array(vals)
            days_data.append({
                'day': i,
                'series': prepare_chart_series(times, vals, 'gsr'),
                'stats': {
                    'min': round(float(arr.min()), 2),
                    'max': round(float(arr.max()), 2),
                    'avg': round(float(arr.mean()), 2),
                },
            })

        return Response({
            'filename': filename,
            'username': username,
            'total_days': len(days_data),
            'days': days_data,
        })

    except Exception as e:
        logger.error(f"view_local_gsr error: {e}")
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def display_csv(request, file_id):
    """
    GET /api/visualization/google-ppg/<file_id>/
    Query params: ?day=1&from=HH:MM&to=HH:MM
    Fetches PPG data from a saved Google Sheet and returns chart JSON.
    """
    sheet = get_object_or_404(GoogleSheet, id=file_id)
    if not request.user.is_superuser and sheet.user != request.user:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(str(settings.GOOGLE_CREDENTIALS_FILE), scope)
        client = gspread.authorize(creds)
        rows = client.open_by_url(sheet.sheet_url).sheet1.get_all_records()
    except Exception as e:
        return Response({'error': f'Google Sheet error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    day_filter = request.GET.get('day')
    from_time = request.GET.get('from')
    to_time = request.GET.get('to')

    day_int = int(day_filter) if day_filter else None
    from_sec = ppg_time_to_seconds(from_time + ':00') if from_time else None
    to_sec = ppg_time_to_seconds(to_time + ':00') if to_time else None

    daily_vals, daily_times = process_csv_into_days(rows, 'PPG', day_filter=day_int, from_sec=from_sec, to_sec=to_sec)

    days_data = []
    for i, (vals, times) in enumerate(zip(daily_vals, daily_times), start=1):
        arr = np.array(vals)
        days_data.append({
            'day': i,
            'series': prepare_chart_series(times, vals, 'ppg'),
            'stats': calculate_ppg_stats(arr),
        })

    return Response({'file_id': file_id, 'total_days': len(days_data), 'days': days_data})


# =============================================================================
# ACTIGRAPHY VIEWS
# =============================================================================

def _load_actigraphy_df(username: str, filename: str) -> tuple[pd.DataFrame | None, Response | None]:
    """Load and validate an actigraphy CSV file. Returns (df, None) or (None, error_response)."""
    file_path = _get_user_file_path(username, filename)
    if file_path is None:
        return None, Response({'error': 'Access denied.'}, status=status.HTTP_403_FORBIDDEN)
    if not os.path.exists(file_path):
        return None, Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)
    try:
        df = _read_csv_with_encoding(file_path)
        return df, None
    except Exception as e:
        return None, Response({'error': f'Error reading file: {e}'}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def homme(request, filename, username=None):
    """
    GET /api/visualization/actigraphy/<filename>/
    Returns a summary of all days (day number + date label + stats).
    """
    if username and not request.user.is_superuser:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)

    target_user = username or request.user.username
    df, err = _load_actigraphy_df(target_user, filename)
    if err:
        return err

    try:
        # Basic column normalization
        df.columns = [c.strip() for c in df.columns]
        time_col = next((c for c in df.columns if 'time' in c.lower()), None)
        if time_col is None:
            return Response({'error': "No 'Time' column found in CSV."}, status=400)

        # Split into days
        df[time_col] = df[time_col].astype(str)
        df['_sec'] = df[time_col].apply(ppg_time_to_seconds)
        df.dropna(subset=['_sec'], inplace=True)

        # Assign day numbers
        day_nums = [1]
        for i in range(1, len(df)):
            if df['_sec'].iloc[i] < df['_sec'].iloc[i-1] - 10:
                day_nums.append(day_nums[-1] + 1)
            else:
                day_nums.append(day_nums[-1])
        df['_day'] = day_nums

        total_days = df['_day'].max()
        days_summary = []
        for d in range(1, total_days + 1):
            day_df = df[df['_day'] == d]
            days_summary.append({
                'day': d,
                'row_count': len(day_df),
                'start_time': day_df[time_col].iloc[0] if len(day_df) else None,
                'end_time': day_df[time_col].iloc[-1] if len(day_df) else None,
            })

        return Response({'filename': filename, 'username': target_user, 'total_days': total_days, 'days': days_summary})

    except Exception as e:
        logger.error(f"homme error: {e}")
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def actigraphy_day_view(request, filename, day_id, username=None):
    """
    GET /api/visualization/actigraphy/<filename>/day/<day_id>/
    Returns all sensor data for a specific day.
    """
    if username and not request.user.is_superuser:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)

    target_user = username or request.user.username
    df, err = _load_actigraphy_df(target_user, filename)
    if err:
        return err

    try:
        df.columns = [c.strip() for c in df.columns]
        time_col = next((c for c in df.columns if 'time' in c.lower()), None)
        if time_col is None:
            return Response({'error': "No 'Time' column found."}, status=400)

        df[time_col] = df[time_col].astype(str)
        df['_sec'] = df[time_col].apply(ppg_time_to_seconds)
        df.dropna(subset=['_sec'], inplace=True)

        day_nums = [1]
        for i in range(1, len(df)):
            if df['_sec'].iloc[i] < df['_sec'].iloc[i-1] - 10:
                day_nums.append(day_nums[-1] + 1)
            else:
                day_nums.append(day_nums[-1])
        df['_day'] = day_nums

        day_df = df[df['_day'] == day_id].drop(columns=['_sec', '_day'])

        if day_df.empty:
            return Response({'error': f'No data for day {day_id}.'}, status=404)

        # Return all available signal columns as series
        numeric_cols = [c for c in day_df.select_dtypes(include=[np.number]).columns]
        series_data = {}
        for col in numeric_cols:
            series_data[col] = prepare_chart_series(
                day_df[time_col].tolist(),
                day_df[col].tolist(),
                col
            )

        return Response({
            'filename': filename,
            'day': day_id,
            'row_count': len(day_df),
            'columns': numeric_cols,
            'series': series_data,
        })

    except Exception as e:
        logger.error(f"actigraphy_day_view error: {e}")
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def actigraphy_stats(request, filename):
    """
    GET /api/visualization/actigraphy-stats/<filename>/
    Returns aggregate statistics per day across all signal columns.
    """
    df, err = _load_actigraphy_df(request.user.username, filename)
    if err:
        return err

    try:
        df.columns = [c.strip() for c in df.columns]
        time_col = next((c for c in df.columns if 'time' in c.lower()), None)
        if not time_col:
            return Response({'error': 'No Time column found.'}, status=400)

        df['_sec'] = df[time_col].astype(str).apply(ppg_time_to_seconds)
        df.dropna(subset=['_sec'], inplace=True)

        day_nums = [1]
        for i in range(1, len(df)):
            if df['_sec'].iloc[i] < df['_sec'].iloc[i-1] - 10:
                day_nums.append(day_nums[-1] + 1)
            else:
                day_nums.append(day_nums[-1])
        df['_day'] = day_nums

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith('_')]
        stats_per_day = []
        for day in df['_day'].unique():
            day_df = df[df['_day'] == day]
            day_stats = {'day': int(day)}
            for col in numeric_cols:
                arr = day_df[col].dropna().values
                if len(arr):
                    day_stats[col] = {
                        'min': round(float(arr.min()), 3),
                        'max': round(float(arr.max()), 3),
                        'avg': round(float(arr.mean()), 3),
                        'std': round(float(arr.std()), 3),
                    }
            stats_per_day.append(day_stats)

        return Response({'filename': filename, 'stats': stats_per_day})

    except Exception as e:
        logger.error(f"actigraphy_stats error: {e}")
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def actigraphy_weekly(request, filename, username=None):
    """
    GET /api/visualization/actigraphy-weekly/<filename>/
    Returns weekly aggregate stats across all days.
    """
    if username and not request.user.is_superuser:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)

    target_user = username or request.user.username
    df, err = _load_actigraphy_df(target_user, filename)
    if err:
        return err

    try:
        df.columns = [c.strip() for c in df.columns]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        weekly_stats = {}
        for col in numeric_cols:
            arr = df[col].dropna().values
            if len(arr):
                weekly_stats[col] = {
                    'min': round(float(arr.min()), 3),
                    'max': round(float(arr.max()), 3),
                    'avg': round(float(arr.mean()), 3),
                    'std': round(float(arr.std()), 3),
                }
        return Response({'filename': filename, 'username': target_user, 'weekly_stats': weekly_stats})

    except Exception as e:
        logger.error(f"actigraphy_weekly error: {e}")
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def actigraphy_day_page(request, filename, day_id, username=None):
    """Alias for actigraphy_day_view (kept for URL compatibility)."""
    return actigraphy_day_view(request, filename=filename, day_id=day_id, username=username)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def compact(request, filename):
    """
    GET /api/visualization/compact/<filename>/
    Returns a compact summary: one data point per minute per signal.
    """
    df, err = _load_actigraphy_df(request.user.username, filename)
    if err:
        return err

    try:
        df.columns = [c.strip() for c in df.columns]
        time_col = next((c for c in df.columns if 'time' in c.lower()), None)
        if not time_col:
            return Response({'error': 'No Time column found.'}, status=400)

        df['_sec'] = df[time_col].astype(str).apply(ppg_time_to_seconds)
        df.dropna(subset=['_sec'], inplace=True)
        df['_min'] = (df['_sec'] // 60).astype(int)

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith('_')]
        compact_data = df.groupby('_min')[numeric_cols].mean().reset_index()

        result = compact_data.to_dict(orient='records')
        return Response({'filename': filename, 'compact': result, 'columns': numeric_cols})

    except Exception as e:
        logger.error(f"compact error: {e}")
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def generate_ppg_graph(request, file_id):
    """
    GET /api/visualization/ppg-graph/<file_id>/
    Returns PPG data from an ExcelFile object.
    """
    from login.models import ExcelFile
    try:
        excel_file = get_object_or_404(ExcelFile, pk=file_id)
        if not request.user.is_superuser and excel_file.user != request.user:
            return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)

        df = pd.read_csv(excel_file.file.path)
        if not {'Time', 'PPG'}.issubset(df.columns):
            return Response({'error': 'CSV must have Time and PPG columns.'}, status=400)

        df = df[['Time', 'PPG']].dropna()
        series = prepare_chart_series(df['Time'].tolist(), df['PPG'].tolist(), 'ppg')
        arr = df['PPG'].values
        return Response({
            'file_id': file_id,
            'series': series,
            'stats': calculate_ppg_stats(arr),
        })
    except Exception as e:
        logger.error(f"generate_ppg_graph error: {e}")
        return Response({'error': str(e)}, status=500)
