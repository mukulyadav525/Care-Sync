"""
HL7 API Views
Handles: HL7 message generation from CSV data, HL7 download (text + PDF)
"""
import os
import csv
import io
import logging
from datetime import datetime

import requests
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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
from login.security import safe_join

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# MSH header for all HL7 messages
HL7_MSH = "MSH|^~\\&|Care-Sync|Facility|HL7Server|HL7Server|{ts}||ORU^R01|{msg_id}|P|2.5|||\n"

SAMPLE_CSV_URL = 'https://raw.githubusercontent.com/sakethwithanh/mHealth-frontend/main/AHLSAM018434.csv'


def _build_hl7_from_records(rows: list) -> str:
    """Build an HL7 message string from a list of record dicts."""
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    msg_id = int(datetime.utcnow().timestamp())
    hl7 = HL7_MSH.format(ts=ts, msg_id=msg_id)

    for record in rows:
        t = record.get('Time', 'unknown')
        sleep = record.get('Sleep', '')
        hl7 += f"PID|1||{t}||{sleep}|||\n"
        if 'GSR' in record:
            hl7 += f"OBX|1|NM|GSR^Galvanic Skin Response^HL7|||{t}|||||AmplitudeData^{t}^Units|{record['GSR']}\n"
        if 'CBT(degC)' in record:
            hl7 += f"OBX|2|NM|CBT^Core Body Temperature^HL7|||{t}|||||AmplitudeData^{t}^Units|{record['CBT(degC)']}\n"
        if 'PPG' in record:
            hl7 += f"OBX|3|NM|PPG^Photoplethysmogram^HL7|||{t}|||||AmplitudeData^{t}^Units|{record['PPG']}\n"
        if 'ECG' in record:
            hl7 += f"OBX|4|NM|ECG^Electrocardiogram^HL7|||{t}|||||AmplitudeData^{t}^Units|{record['ECG']}\n"
    return hl7


def _build_hl7_from_raw_df(df: pd.DataFrame) -> str:
    """Build HL7 from a raw device CSV with IMU/PPG columns."""
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    msg_id = int(datetime.utcnow().timestamp())
    hl7 = HL7_MSH.format(ts=ts, msg_id=msg_id)

    for _, row in df.iterrows():
        t = row.get('Time', 'unknown')
        hl7 += f"PID|1||{t}||||\n"
        field_map = {
            'GSR': 'GSR^Galvanic Skin Response^HL7',
            'PPG': 'PPG^Photoplethysmogram^HL7',
            'AccelX': 'ACCEL_X^Accelerometer X^HL7',
            'AccelY': 'ACCEL_Y^Accelerometer Y^HL7',
            'AccelZ': 'ACCEL_Z^Accelerometer Z^HL7',
            'GyroX': 'GYRO_X^Gyroscope X^HL7',
            'GyroY': 'GYRO_Y^Gyroscope Y^HL7',
            'GyroZ': 'GYRO_Z^Gyroscope Z^HL7',
        }
        for i, (col, label) in enumerate(field_map.items(), start=1):
            if col in row and pd.notna(row[col]):
                hl7 += f"OBX|{i}|NM|{label}|||{t}|||||AmplitudeData^{t}^Units|{row[col]}\n"
    return hl7


def _fetch_sample_csv() -> list:
    """Download and parse the sample CSV from GitHub."""
    resp = requests.get(SAMPLE_CSV_URL, timeout=15)
    resp.raise_for_status()
    reader = csv.DictReader(resp.text.splitlines())
    return list(reader)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def generate_hl7(request):
    """
    GET /api/hl7/generate/
    Generates an HL7 message from the sample CSV and returns it as plain text.
    """
    try:
        rows = _fetch_sample_csv()
        hl7 = _build_hl7_from_records(rows)
        return HttpResponse(hl7, content_type='text/plain')
    except Exception as e:
        logger.error(f"HL7 generation error: {e}")
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def download_hl7(request):
    """
    GET /api/hl7/download/
    Downloads the HL7 message as a .hl7 attachment.
    """
    try:
        rows = _fetch_sample_csv()
        hl7 = _build_hl7_from_records(rows)
        response = HttpResponse(hl7, content_type='text/plain')
        response['Content-Disposition'] = 'attachment; filename="hl7_messages.hl7"'
        return response
    except Exception as e:
        logger.error(f"HL7 download error: {e}")
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def view_hl7(request, file_id):
    """
    GET /api/hl7/view/<file_id>/
    Generates HL7 from a saved Google Sheet and returns as JSON with message string.
    """
    google_sheet = get_object_or_404(GoogleSheet, id=file_id)

    if not request.user.is_superuser and google_sheet.user != request.user:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            str(settings.GOOGLE_CREDENTIALS_FILE), scope
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_url(google_sheet.sheet_url).sheet1
        rows = sheet.get_all_records()
    except Exception as e:
        logger.error(f"Google Sheet access error: {e}")
        return Response({'error': f'Error accessing Google Sheet: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    hl7 = _build_hl7_from_records(rows)
    return Response({'file_id': file_id, 'hl7_message': hl7})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def convert_local_csv_to_hl7(request, filename):
    """
    GET /api/hl7/local/<filename>/
    Converts the user's local CSV to HL7 text and returns it.
    """
    username = request.user.username
    base_dir = str(settings.USER_FILES_BASE_DIR)
    file_path = safe_join(base_dir, username, filename)

    # Security: prevent directory traversal
    if file_path is None:
        return Response({'error': 'Access denied.'}, status=status.HTTP_403_FORBIDDEN)

    if not os.path.exists(file_path):
        return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

    try:
        df = pd.read_csv(file_path)

        required_cols = {'Hour', 'Minute', 'Second', 'Millisecond', 'Red', 'IR', 'GSR'}
        if not required_cols.issubset(df.columns):
            return Response(
                {'error': 'CSV missing required columns: Hour, Minute, Second, Millisecond, Red, IR, GSR'},
                status=status.HTTP_400_BAD_REQUEST
            )

        df['Time'] = df.apply(
            lambda r: f"{int(r['Hour']):02}:{int(r['Minute']):02}:{int(r['Second']):02}.{int(r['Millisecond']):03}",
            axis=1
        )
        df['PPG'] = df.apply(lambda r: r['Red'] / r['IR'] if r['IR'] != 0 else None, axis=1)
        df.dropna(inplace=True)

        hl7 = _build_hl7_from_raw_df(df)
        return Response({'filename': filename, 'hl7_message': hl7})

    except Exception as e:
        logger.error(f"CSV-to-HL7 error: {e}")
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def download_hl7_pdf(request, filename):
    """
    GET /api/hl7/pdf/<filename>/
    Converts the user's local CSV to HL7 and returns as a downloadable PDF.
    """
    username = request.user.username
    base_dir = str(settings.USER_FILES_BASE_DIR)
    file_path = safe_join(base_dir, username, filename)

    if file_path is None:
        return Response({'error': 'Access denied.'}, status=status.HTTP_403_FORBIDDEN)

    if not os.path.exists(file_path):
        return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

    try:
        df = pd.read_csv(file_path)

        time_cols = {'Hour', 'Minute', 'Second', 'Millisecond'}
        if time_cols.issubset(df.columns):
            df['Time'] = df.apply(
                lambda r: f"{int(r['Hour']):02}:{int(r['Minute']):02}:{int(r['Second']):02}.{int(r['Millisecond']):03}",
                axis=1
            )
        if {'Red', 'IR'}.issubset(df.columns):
            df['PPG'] = df.apply(lambda r: r['Red'] / r['IR'] if r['IR'] != 0 else None, axis=1)

        df.dropna(inplace=True)
        hl7 = _build_hl7_from_raw_df(df)

        # Render PDF
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}.pdf"'

        pdf = canvas.Canvas(response, pagesize=letter)
        pdf.setFont("Helvetica", 9)
        y = 750
        for line in hl7.split('\n'):
            pdf.drawString(50, y, line[:120])  # Truncate long lines
            y -= 14
            if y < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 9)
                y = 750

        pdf.save()
        return response

    except Exception as e:
        logger.error(f"HL7 PDF error: {e}")
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
