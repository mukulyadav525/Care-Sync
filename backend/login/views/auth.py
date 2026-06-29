"""
Authentication API Views
Handles: Signup (email OTP), Login with 2FA (email OTP), Logout (token revoke),
         Change password.

Security notes:
  * OTP codes are stored only as salted hashes (EmailOTP), expire after 10 min,
    and are limited to a few attempts.
  * Signup never stores a plaintext password — the pending account holds an
    already-hashed password until the OTP is verified.
  * Login is two-factor: password first, then an emailed one-time code.
  * Sensitive endpoints are IP rate-limited via scoped throttles.
"""
import os
import secrets
import logging

from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.mail import send_mail
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.conf import settings

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from rest_framework_simplejwt.tokens import RefreshToken

from login.models import EmailOTP, UserProfile
from login.security import AuthRateThrottle, OTPRateThrottle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {'refresh': str(refresh), 'access': str(refresh.access_token)}


def _user_public(user):
    return {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'is_superuser': user.is_superuser,
    }


def _generate_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _mask_email(email: str) -> str:
    try:
        name, domain = email.split('@', 1)
        head = name[0] if name else ''
        return f"{head}{'*' * max(1, len(name) - 1)}@{domain}"
    except ValueError:
        return email


def _send_otp_email(email: str, code: str, purpose: str) -> bool:
    label = 'sign-in' if purpose == EmailOTP.PURPOSE_LOGIN else 'sign-up'
    try:
        send_mail(
            subject='Your Care-Sync verification code',
            message=(f'Your Care-Sync {label} verification code is: {code}\n\n'
                     'It expires in 10 minutes. If you did not request this, ignore this email.'),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to send OTP email to {_mask_email(email)}: {e}")
        return False


def _issue_otp(email: str, purpose: str, *, username='', password_hash='') -> str:
    """Create/replace an OTP for (email, purpose) and return the raw code."""
    EmailOTP.objects.filter(email=email, purpose=purpose).delete()
    code = _generate_code()
    otp = EmailOTP(email=email, purpose=purpose, username=username, password_hash=password_hash)
    otp.set_code(code)
    otp.save()
    return code


def _consume_otp(email: str, purpose: str, entered_code: str):
    """
    Validate an OTP. Returns (otp_obj, None) on success or (None, error_response).
    On success the caller is responsible for deleting the OTP.
    """
    otp = EmailOTP.objects.filter(email=email, purpose=purpose).order_by('-created_at').first()
    if otp is None:
        return None, Response({'error': 'No verification in progress. Please start again.'},
                              status=status.HTTP_400_BAD_REQUEST)
    if otp.is_expired():
        otp.delete()
        return None, Response({'error': 'Code expired. Please request a new one.'},
                              status=status.HTTP_400_BAD_REQUEST)
    if otp.attempts >= EmailOTP.MAX_ATTEMPTS:
        otp.delete()
        return None, Response({'error': 'Too many attempts. Please start again.'},
                              status=status.HTTP_429_TOO_MANY_REQUESTS)
    if not otp.check_code(str(entered_code).strip()):
        otp.attempts += 1
        otp.save(update_fields=['attempts'])
        return None, Response({'error': 'Invalid code. Please try again.'},
                              status=status.HTTP_400_BAD_REQUEST)
    return otp, None


def _otp_response(payload: dict, code: str) -> Response:
    if settings.OTP_DEBUG_RETURN:
        payload = {**payload, 'dev_otp': code}
    return Response(payload, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Signup
# ---------------------------------------------------------------------------

@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([AuthRateThrottle])
def signup(request):
    """
    POST /api/auth/signup/  Body: { username, email, password1, password2 }
    Validates the account and emails a verification code (no account is created
    until the code is verified).
    """
    data = request.data
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password1 = data.get('password1', '')
    password2 = data.get('password2', '')

    if not all([username, email, password1, password2]):
        return Response({'error': 'All fields are required.'}, status=status.HTTP_400_BAD_REQUEST)
    try:
        validate_email(email)
    except ValidationError:
        return Response({'error': 'Invalid email address.'}, status=status.HTTP_400_BAD_REQUEST)
    if password1 != password2:
        return Response({'error': 'Passwords do not match.'}, status=status.HTTP_400_BAD_REQUEST)
    if User.objects.filter(username__iexact=username).exists():
        return Response({'error': 'Username is already taken.'}, status=status.HTTP_409_CONFLICT)
    if User.objects.filter(email__iexact=email).exists():
        return Response({'error': 'Email is already registered.'}, status=status.HTTP_409_CONFLICT)

    # Enforce password strength using the configured Django validators
    try:
        validate_password(password1)
    except ValidationError as e:
        return Response({'error': ' '.join(e.messages)}, status=status.HTTP_400_BAD_REQUEST)

    from django.contrib.auth.hashers import make_password
    code = _issue_otp(email, EmailOTP.PURPOSE_SIGNUP,
                      username=username, password_hash=make_password(password1))
    sent = _send_otp_email(email, code, EmailOTP.PURPOSE_SIGNUP)
    if not sent and not settings.OTP_DEBUG_RETURN:
        return Response({'error': 'Failed to send verification email. Please try again.'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return _otp_response({'message': 'Verification code sent to your email.',
                          'email': _mask_email(email)}, code)


@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([OTPRateThrottle])
def send_otp(request):
    """POST /api/auth/send-otp/  Body: { email } — resend the signup code."""
    email = request.data.get('email', '').strip()
    otp = EmailOTP.objects.filter(email=email, purpose=EmailOTP.PURPOSE_SIGNUP).order_by('-created_at').first()
    if otp is None:
        return Response({'error': 'No signup in progress for this email.'}, status=status.HTTP_400_BAD_REQUEST)

    code = _generate_code()
    otp.set_code(code)
    otp.attempts = 0
    otp.save(update_fields=['code_hash', 'attempts'])
    sent = _send_otp_email(email, code, EmailOTP.PURPOSE_SIGNUP)
    if not sent and not settings.OTP_DEBUG_RETURN:
        return Response({'error': 'Failed to send code.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return _otp_response({'message': 'A new code was sent.'}, code)


@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([OTPRateThrottle])
def verify_otp(request):
    """
    POST /api/auth/verify-otp/  Body: { email, otp }
    Verifies the signup code, creates the account, returns JWT tokens.
    """
    email = request.data.get('email', '').strip()
    entered = request.data.get('otp', '')

    otp, err = _consume_otp(email, EmailOTP.PURPOSE_SIGNUP, entered)
    if err:
        return err

    username = otp.username
    # Final uniqueness guard (race / replay)
    if User.objects.filter(username__iexact=username).exists() or \
       User.objects.filter(email__iexact=email).exists():
        otp.delete()
        return Response({'error': 'Account already exists. Please sign in.'}, status=status.HTTP_409_CONFLICT)

    try:
        user = User(username=username, email=email)
        user.password = otp.password_hash  # already a valid hashed password
        user.save()
        UserProfile.objects.get_or_create(user=user)
        os.makedirs(os.path.join(settings.USER_FILES_BASE_DIR, username), exist_ok=True)
        otp.delete()
    except Exception as e:
        logger.error(f"User creation failed: {e}")
        return Response({'error': 'Account creation failed. Please try again.'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({'message': 'Account created successfully.', 'user': _user_public(user),
                     **get_tokens_for_user(user)}, status=status.HTTP_201_CREATED)


# ---------------------------------------------------------------------------
# Login (two-factor)
# ---------------------------------------------------------------------------

@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([AuthRateThrottle])
def login_view(request):
    """
    POST /api/auth/login/  Body: { username, password }
    Step 1 of 2FA: verifies the password, then emails a one-time code.
    Returns { otp_required: true } — no tokens are issued yet.
    """
    username = request.data.get('username', '').strip()
    password = request.data.get('password', '')
    if not username or not password:
        return Response({'error': 'Username and password are required.'}, status=status.HTTP_400_BAD_REQUEST)

    user = authenticate(request, username=username, password=password)
    if user is None:
        return Response({'error': 'Invalid username or password.'}, status=status.HTTP_401_UNAUTHORIZED)
    if not user.email:
        return Response({'error': 'This account has no email set for two-factor authentication.'},
                        status=status.HTTP_400_BAD_REQUEST)

    code = _issue_otp(user.email, EmailOTP.PURPOSE_LOGIN, username=user.username)
    sent = _send_otp_email(user.email, code, EmailOTP.PURPOSE_LOGIN)
    if not sent and not settings.OTP_DEBUG_RETURN:
        return Response({'error': 'Failed to send verification email. Please try again.'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return _otp_response({'otp_required': True, 'username': user.username,
                          'email': _mask_email(user.email)}, code)


@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([OTPRateThrottle])
def verify_login_otp(request):
    """
    POST /api/auth/login/verify/  Body: { username, otp }
    Step 2 of 2FA: verifies the emailed code and issues JWT tokens.
    """
    username = request.data.get('username', '').strip()
    entered = request.data.get('otp', '')

    user = User.objects.filter(username__iexact=username).first()
    if user is None or not user.email:
        return Response({'error': 'Invalid sign-in attempt.'}, status=status.HTTP_400_BAD_REQUEST)

    otp, err = _consume_otp(user.email, EmailOTP.PURPOSE_LOGIN, entered)
    if err:
        return err
    otp.delete()

    return Response({'message': 'Login successful.', 'user': _user_public(user),
                     **get_tokens_for_user(user)}, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """POST /api/auth/logout/  Body: { refresh } — revokes the refresh token."""
    refresh_token = request.data.get('refresh')
    if not refresh_token:
        return Response({'error': 'Refresh token is required.'}, status=status.HTTP_400_BAD_REQUEST)
    try:
        RefreshToken(refresh_token).blacklist()
    except Exception:
        return Response({'error': 'Invalid or already revoked token.'}, status=status.HTTP_400_BAD_REQUEST)
    return Response({'message': 'Logged out successfully.'}, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password(request):
    """
    POST /api/auth/change-password/  Body: { old_password, new_password }
    """
    old_password = request.data.get('old_password', '')
    new_password = request.data.get('new_password', '')
    user = request.user

    if not user.check_password(old_password):
        return Response({'error': 'Current password is incorrect.'}, status=status.HTTP_400_BAD_REQUEST)
    try:
        validate_password(new_password, user=user)
    except ValidationError as e:
        return Response({'error': ' '.join(e.messages)}, status=status.HTTP_400_BAD_REQUEST)

    user.set_password(new_password)
    user.save(update_fields=['password'])
    return Response({'message': 'Password updated successfully.'}, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def me(request):
    """GET /api/auth/me/ - current user."""
    user = request.user
    return Response({**_user_public(user), 'date_joined': user.date_joined})
