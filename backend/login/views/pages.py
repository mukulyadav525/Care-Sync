"""
Pages / General API Views
Provides metadata endpoints for pages that used to be template-rendered.
All static page views now return JSON metadata or user profile data.
"""
import logging
import os
from collections import Counter
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.models import User
from django.utils import timezone

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response

from login.models import ContactMessage, Device, AlertFired, HRVAnomalyAlert, UploadedDocument

logger = logging.getLogger(__name__)


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """GET /api/health/ - Simple health check endpoint."""
    return Response({'status': 'ok', 'service': 'Care-Sync API'})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile(request):
    """
    GET /api/profile/
    Returns the authenticated user's profile data.
    """
    user = request.user
    profile_data = {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'is_superuser': user.is_superuser,
        'date_joined': user.date_joined,
        'form_submitted': False,
    }
    try:
        profile_data['form_submitted'] = user.userprofile.form_submitted
    except Exception:
        pass
    return Response(profile_data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def admin_patients(request):
    """
    GET /api/admin/patients/
    Superuser-only: list of all users with session counts and consent status.
    """
    if not request.user.is_superuser:
        return Response({'error': 'Admin access required.'}, status=status.HTTP_403_FORBIDDEN)

    base_dir = str(settings.USER_FILES_BASE_DIR)
    users = User.objects.filter(is_superuser=False).select_related('userprofile').order_by('username')

    patients = []
    for u in users:
        # Count sessions (directories under Users/<username>/)
        user_dir = os.path.join(base_dir, u.username)
        session_count = 0
        last_session = None
        if os.path.isdir(user_dir):
            dirs = [
                os.path.join(user_dir, d) for d in os.listdir(user_dir)
                if os.path.isdir(os.path.join(user_dir, d))
            ]
            session_count = len(dirs)
            if dirs:
                latest = max(dirs, key=os.path.getmtime)
                import datetime
                last_session = datetime.datetime.fromtimestamp(os.path.getmtime(latest)).isoformat()

        try:
            consent = u.userprofile.form_submitted
        except Exception:
            consent = False

        patients.append({
            'username': u.username,
            'email': u.email,
            'date_joined': u.date_joined.isoformat(),
            'session_count': session_count,
            'last_session': last_session,
            'consent': consent,
        })

    return Response({'patients': patients})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def admin_overview(request):
    """
    GET /api/admin/overview/
    Superuser-only: platform-wide analytics — adoption, alert volume, and a
    crude population health signal derived from HRV anomaly alert reasons.
    Deliberately avoids re-reading every user's raw signal CSVs (expensive);
    everything here comes from cheap DB aggregates plus a directory count.
    """
    if not request.user.is_superuser:
        return Response({'error': 'Admin access required.'}, status=status.HTTP_403_FORBIDDEN)

    base_dir = str(settings.USER_FILES_BASE_DIR)
    users = User.objects.filter(is_superuser=False).select_related('userprofile')

    total_sessions = 0
    users_with_sessions = 0
    users_with_consent = 0
    for u in users:
        user_dir = os.path.join(base_dir, u.username)
        n = 0
        if os.path.isdir(user_dir):
            n = len([d for d in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, d))])
        total_sessions += n
        if n > 0:
            users_with_sessions += 1
        try:
            if u.userprofile.form_submitted:
                users_with_consent += 1
        except Exception:
            pass

    devices = Device.objects.all()
    devices_online = sum(1 for d in devices if d.is_online)

    doc_counts = Counter(UploadedDocument.objects.values_list('doc_type', flat=True))

    cutoff_7d = timezone.now() - timedelta(days=7)
    threshold_alerts_7d = AlertFired.objects.filter(fired_at__gte=cutoff_7d).count()
    hrv_alerts_7d = HRVAnomalyAlert.objects.filter(created_at__gte=cutoff_7d)
    hrv_watch_7d = hrv_alerts_7d.filter(severity='watch').count()
    hrv_alert_7d = hrv_alerts_7d.filter(severity='alert').count()

    # Crude population health signal: which reason phrases show up most often
    # in HRV anomaly alerts over the last 30 days, across all users. Reasons
    # are free-text ("HR way above expected", "RMSSD dropped 35%", ...) so
    # this buckets on the leading phrase rather than trying to parse them.
    cutoff_30d = timezone.now() - timedelta(days=30)
    reason_counter = Counter()
    for reasons_text in HRVAnomalyAlert.objects.filter(created_at__gte=cutoff_30d).values_list('reasons', flat=True):
        for line in (reasons_text or '').split('\n'):
            line = line.strip()
            if line and line != 'No significant deviation':
                # bucket by the phrase category, not exact numbers
                key = line.split(' above')[0].split(' below')[0].split(' dropped')[0].split(' elevated')[0].split(':')[0][:40]
                reason_counter[key] += 1

    recent_hrv_alerts = HRVAnomalyAlert.objects.select_related('user')[:10]

    return Response({
        'users': {
            'total': users.count(),
            'with_sessions': users_with_sessions,
            'with_consent': users_with_consent,
        },
        'sessions': {'total': total_sessions},
        'devices': {'total': devices.count(), 'online': devices_online},
        'documents_by_type': dict(doc_counts),
        'alerts_last_7d': {
            'threshold': threshold_alerts_7d,
            'hrv_watch': hrv_watch_7d,
            'hrv_alert': hrv_alert_7d,
        },
        'top_alert_reasons_30d': [{'reason': r, 'count': c} for r, c in reason_counter.most_common(8)],
        'recent_hrv_alerts': [
            {
                'id': a.id, 'user': a.user.username, 'owner': a.owner, 'session': a.session,
                'severity': a.severity, 'score': a.score, 'emailed': a.emailed,
                'created_at': a.created_at.isoformat(),
            }
            for a in recent_hrv_alerts
        ],
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def contact_view(request):
    """
    POST /api/contact/
    Body: { name, email, subject, message }
    Saves a contact message.
    """
    name = request.data.get('name', '').strip()
    email = request.data.get('email', '').strip()
    subject = request.data.get('subject', '').strip()
    message = request.data.get('message', '').strip()

    if not all([name, email, message]):
        return Response({'error': 'Name, email, and message are required.'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        ContactMessage.objects.create(name=name, email=email, subject=subject, message=message)
        return Response({'message': 'Your message has been sent. Thank you!'}, status=status.HTTP_201_CREATED)
    except Exception as e:
        logger.error(f"Contact form error: {e}")
        return Response({'error': 'Failed to save message.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
