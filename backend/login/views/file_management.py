"""
File Management API Views
Handles: Local file listing, file detail, Google Sheets upload/view
"""
import os
import re
import pickle
import logging
import mimetypes

from datetime import datetime

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from django.conf import settings
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, parser_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from login.models import GoogleSheet, ExcelFile, UploadedDocument
from login.security import safe_join, UploadRateThrottle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_user_files_dir(username: str) -> str:
    """Return the absolute path to a user's file directory, creating it if needed."""
    user_dir = os.path.join(settings.USER_FILES_BASE_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def _file_stat(file_path: str, filename: str, doc_type: str = 'other') -> dict:
    """Return a dict of file metadata."""
    stat = os.stat(file_path)
    return {
        'name': filename,
        'size': stat.st_size,
        'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'type': mimetypes.guess_type(file_path)[0] or 'application/octet-stream',
        'doc_type': doc_type,
    }


def _doc_types_for(username: str) -> dict:
    """filename -> doc_type map for one user's UploadedDocument rows."""
    return dict(
        UploadedDocument.objects.filter(user__username=username).values_list('filename', 'doc_type')
    )


# ---------------------------------------------------------------------------
# Local File Views
# ---------------------------------------------------------------------------

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def local_files_view(request):
    """
    GET /api/files/local/
    Lists local files for the authenticated user.
    Superusers see all users' files.
    """
    base_dir = str(settings.USER_FILES_BASE_DIR)

    if request.user.is_superuser:
        users_data = []
        if os.path.exists(base_dir):
            for uname in os.listdir(base_dir):
                user_folder = os.path.join(base_dir, uname)
                if not os.path.isdir(user_folder):
                    continue
                doc_types = _doc_types_for(uname)
                files = [
                    _file_stat(os.path.join(user_folder, f), f, doc_types.get(f, 'other'))
                    for f in os.listdir(user_folder)
                    if os.path.isfile(os.path.join(user_folder, f))
                ]
                users_data.append({'username': uname, 'files': files})
        return Response({'is_superuser': True, 'users': users_data, 'doc_type_choices': UploadedDocument.DOC_TYPE_CHOICES})
    else:
        user_folder = _get_user_files_dir(request.user.username)
        doc_types = _doc_types_for(request.user.username)
        files = [
            _file_stat(os.path.join(user_folder, f), f, doc_types.get(f, 'other'))
            for f in os.listdir(user_folder)
            if os.path.isfile(os.path.join(user_folder, f))
        ]
        return Response({
            'is_superuser': False,
            'username': request.user.username,
            'files': files,
            'doc_type_choices': UploadedDocument.DOC_TYPE_CHOICES,
        })


ALLOWED_EXTENSIONS = {'.csv', '.txt', '.json', '.ppg', '.edf', '.xml', '.pdf', '.xlsx', '.xls'}
MAX_UPLOAD_MB = 50


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@throttle_classes([UploadRateThrottle])
@parser_classes([MultiPartParser, FormParser])
def upload_local_file(request):
    """
    POST /api/files/local/upload/
    Multipart upload of a file into the user's local directory.
    """
    uploaded = request.FILES.get('file')
    if not uploaded:
        return Response({'error': 'No file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return Response({'error': f'File type {ext} not allowed. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}, status=status.HTTP_400_BAD_REQUEST)

    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        return Response({'error': f'File exceeds {MAX_UPLOAD_MB} MB limit.'}, status=status.HTTP_400_BAD_REQUEST)

    doc_type = request.data.get('doc_type', 'other')
    valid_doc_types = {c[0] for c in UploadedDocument.DOC_TYPE_CHOICES}
    if doc_type not in valid_doc_types:
        doc_type = 'other'

    user_folder = _get_user_files_dir(request.user.username)

    # Sanitize filename — strip path components, keep only the basename
    safe_name = os.path.basename(uploaded.name).strip()
    safe_name = re.sub(r'[^\w.\-]', '_', safe_name)
    if not safe_name:
        return Response({'error': 'Invalid filename.'}, status=status.HTTP_400_BAD_REQUEST)

    dest = os.path.join(user_folder, safe_name)

    with open(dest, 'wb') as f:
        for chunk in uploaded.chunks():
            f.write(chunk)

    UploadedDocument.objects.update_or_create(
        user=request.user, filename=safe_name, defaults={'doc_type': doc_type}
    )

    return Response(_file_stat(dest, safe_name, doc_type), status=status.HTTP_201_CREATED)


@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def update_document_type(request, username, filename):
    """
    PATCH /api/files/local/<username>/<filename>/type/
    Body: { doc_type }
    Re-categorizes an already-uploaded file (e.g. move it from "Other" to
    "Lab Report") without re-uploading.
    """
    if not request.user.is_superuser and username != request.user.username:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)

    base_dir = str(settings.USER_FILES_BASE_DIR)
    file_path = safe_join(base_dir, username, filename)
    if file_path is None:
        return Response({'error': 'Access denied.'}, status=status.HTTP_403_FORBIDDEN)
    if not os.path.isfile(file_path):
        return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

    doc_type = request.data.get('doc_type')
    valid_doc_types = {c[0] for c in UploadedDocument.DOC_TYPE_CHOICES}
    if doc_type not in valid_doc_types:
        return Response({'error': f'doc_type must be one of {sorted(valid_doc_types)}'}, status=status.HTTP_400_BAD_REQUEST)

    owner = get_object_or_404(User, username=username)
    UploadedDocument.objects.update_or_create(user=owner, filename=filename, defaults={'doc_type': doc_type})

    return Response({'name': filename, 'doc_type': doc_type})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def view_file_data(request, username, filename):
    """
    GET /api/files/local/<username>/<filename>/
    Returns metadata and text content of a specific local file.
    """
    base_dir = str(settings.USER_FILES_BASE_DIR)

    # Permission check
    if not request.user.is_superuser and username != request.user.username:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)

    file_path = safe_join(base_dir, username, filename)

    # Security: prevent directory traversal
    if file_path is None:
        return Response({'error': 'Access denied.'}, status=status.HTTP_403_FORBIDDEN)

    if not os.path.isfile(file_path):
        return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

    file_info = _file_stat(file_path, filename)
    file_info['owner'] = username

    # Read text content if possible
    mime_type = file_info['type']
    if mime_type and (mime_type.startswith('text/') or mime_type == 'application/json'):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                file_info['content'] = f.read()
        except Exception:
            file_info['content'] = None
    else:
        file_info['content'] = None
        file_info['binary'] = True

    # Flag CSVs/PPG files for visualization links
    if filename.lower().endswith(('.ppg', '.csv')):
        file_info['has_ppg'] = True
        file_info['ppg_url'] = f'/api/visualization/local-ppg/{filename}/'
        file_info['gsr_url'] = f'/api/visualization/local-gsr/{filename}/'

    return Response(file_info)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_local_file(request, username, filename):
    """
    DELETE /api/files/local/<username>/<filename>/delete/
    Deletes a local file (and its document-category record, if any).
    """
    base_dir = str(settings.USER_FILES_BASE_DIR)

    if not request.user.is_superuser and username != request.user.username:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)

    file_path = safe_join(base_dir, username, filename)

    if file_path is None:
        return Response({'error': 'Access denied.'}, status=status.HTTP_403_FORBIDDEN)

    if not os.path.isfile(file_path):
        return Response({'error': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

    os.remove(file_path)
    UploadedDocument.objects.filter(user__username=username, filename=filename).delete()
    return Response({'message': f'{filename} deleted successfully.'})


# ---------------------------------------------------------------------------
# Google Sheet Views
# ---------------------------------------------------------------------------

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def view_google_sheets(request):
    """
    GET /api/files/sheets/
    Lists Google Sheets for the current user (or all for superusers).
    """
    if request.user.is_superuser:
        sheets = GoogleSheet.objects.select_related('user').all().values(
            'id', 'title', 'sheet_url', 'user__username'
        )
    else:
        sheets = GoogleSheet.objects.filter(user=request.user).values(
            'id', 'title', 'sheet_url'
        )
    return Response(list(sheets))


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_google_sheet(request):
    """
    POST /api/files/sheets/
    Body: { title, google_sheet_url }
    Saves a Google Sheet reference for the user.
    """
    title = request.data.get('title', '').strip()
    sheet_url = request.data.get('google_sheet_url', '').strip()

    if not title or not sheet_url:
        return Response({'error': 'Title and Google Sheet URL are required.'}, status=status.HTTP_400_BAD_REQUEST)

    sheet = GoogleSheet(user=request.user, title=title, sheet_url=sheet_url)
    sheet.save()
    return Response({'message': 'Google Sheet saved.', 'id': sheet.id}, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def google_sheet_detail(request, file_id):
    """
    GET /api/files/sheets/<file_id>/
    Returns sheet URL so the frontend can redirect.
    """
    sheet = get_object_or_404(GoogleSheet, id=file_id)
    if not request.user.is_superuser and sheet.user != request.user:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)
    return Response({'id': sheet.id, 'title': sheet.title, 'url': sheet.sheet_url})


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_google_sheet(request, file_id):
    """
    DELETE /api/files/sheets/<file_id>/
    """
    sheet = get_object_or_404(GoogleSheet, id=file_id)
    if not request.user.is_superuser and sheet.user != request.user:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)
    sheet.delete()
    return Response({'message': 'Sheet deleted.'})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def form_submit(request):
    """
    POST /api/files/form-submit/
    Appends the submitted health consent form as a new row in the consent Google Sheet.
    """
    creds_file = str(settings.GOOGLE_CREDENTIALS_FILE)
    google_sheet_url = "https://docs.google.com/spreadsheets/d/1ikw_LmPFTgLVHNcSthCZlKdAf9JZVnHb-bTLCq9_6FI/edit?usp=sharing"
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    data = request.data
    row = [
        request.user.username,
        data.get('age', ''),
        data.get('gender', ''),
        data.get('height', ''),
        data.get('weight', ''),
        data.get('respiratory_conditions', ''),
        data.get('cardiovascular_conditions', ''),
        data.get('cardiovascular_symptoms', ''),
        data.get('metabolic_conditions', ''),
        data.get('mental_health_conditions', ''),
        data.get('stress_level', ''),
        data.get('lifestyle_factors', ''),
        data.get('sleep_hours', ''),
        data.get('sleep_disorders', ''),
        data.get('last_medical_checkup', ''),
        data.get('health_concerns', ''),
        data.get('consent_data', ''),
        data.get('consent_participate', ''),
        data.get('consent_withdraw', ''),
        datetime.now().isoformat(),
    ]

    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(google_sheet_url).sheet1
        sheet.append_row(row)

        try:
            user_profile = request.user.userprofile
            user_profile.form_submitted = True
            user_profile.save()
        except Exception:
            pass

        return Response({'verified': True, 'message': 'Consent form submitted successfully.'})

    except Exception as e:
        logger.exception("Google Sheet access error")
        return Response({'error': f'Error accessing Google Sheet: {type(e).__name__}: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
