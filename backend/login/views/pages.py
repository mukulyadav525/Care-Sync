"""
Pages / General API Views
Provides metadata endpoints for pages that used to be template-rendered.
All static page views now return JSON metadata or user profile data.
"""
import logging
import os

from django.conf import settings
from django.contrib.auth.models import User

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response

from login.models import ContactMessage

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
