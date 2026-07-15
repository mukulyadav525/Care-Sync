"""
Shared security helpers for the login app.
"""
import os

from rest_framework.throttling import AnonRateThrottle, UserRateThrottle


def safe_join(base: str, *parts: str) -> str | None:
    """
    Safely resolve a path under ``base``.

    Returns the absolute, symlink-resolved path if it stays inside ``base``,
    otherwise ``None``. Protects against directory traversal (``..``) and
    absolute-path injection in user-supplied path segments.
    """
    base_real = os.path.realpath(base)
    target = os.path.realpath(os.path.join(base_real, *parts))
    if target == base_real or target.startswith(base_real + os.sep):
        return target
    return None


# --- Scoped throttles for sensitive, mostly-anonymous endpoints -------------

class AuthRateThrottle(AnonRateThrottle):
    """Throttle login / signup attempts by client IP (rate: 'auth')."""
    scope = 'auth'


class OTPRateThrottle(AnonRateThrottle):
    """Throttle OTP verification attempts by client IP (rate: 'otp')."""
    scope = 'otp'


class HeartbeatRateThrottle(AnonRateThrottle):
    """Throttle device heartbeats by client IP (rate: 'heartbeat')."""
    scope = 'heartbeat'


class UploadRateThrottle(UserRateThrottle):
    """Throttle file uploads per authenticated user (rate: 'upload').
    Uploads are always authenticated (IsAuthenticated), so this must be
    UserRateThrottle, not AnonRateThrottle — DRF's AnonRateThrottle skips
    throttling entirely once request.user is authenticated (its cache key
    is None for authenticated requests), which would make it a no-op here."""
    scope = 'upload'
