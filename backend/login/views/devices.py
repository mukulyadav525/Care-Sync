"""
Connected-device registry & heartbeat API.

  GET    /api/devices/                list the caller's devices (+ live status)
  POST   /api/devices/register/       register a device, returns its secret key once
  DELETE /api/devices/<pk>/delete/    remove a device
  POST   /api/devices/heartbeat/      device check-in (authenticated by X-Device-Key)

The heartbeat endpoint is not behind JWT (devices aren't users) but still
requires a valid per-device secret key, so it is never anonymous/free.
"""
import logging

from django.utils import timezone

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response

from login.models import Device
from login.security import HeartbeatRateThrottle

logger = logging.getLogger(__name__)


def _serialize(device: Device, include_key: bool = False) -> dict:
    data = {
        'id': device.id,
        'device_id': device.device_id,
        'name': device.name,
        'firmware': device.firmware,
        'battery': device.battery,
        'current_session': device.current_session,
        'last_seen': device.last_seen.isoformat() if device.last_seen else None,
        'is_online': device.is_online,
        'owner': device.user.username,
        'created_at': device.created_at.isoformat(),
    }
    if include_key:
        data['key'] = device.key
    return data


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_devices(request):
    """List the caller's devices (all devices for superusers)."""
    qs = Device.objects.select_related('user')
    if not request.user.is_superuser:
        qs = qs.filter(user=request.user)
    devices = [_serialize(d) for d in qs.order_by('-last_seen', 'name')]
    online = sum(1 for d in devices if d['is_online'])
    return Response({'devices': devices, 'online': online, 'total': len(devices)})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def register_device(request):
    """
    Register a new device. Returns the secret key ONCE — it must be copied into
    the device/ingestion config and is not retrievable later.
    """
    device_id = request.data.get('device_id', '').strip()
    name = request.data.get('name', '').strip() or 'Wearable'
    if not device_id:
        return Response({'error': 'device_id is required.'}, status=status.HTTP_400_BAD_REQUEST)
    if Device.objects.filter(user=request.user, device_id=device_id).exists():
        return Response({'error': 'A device with this ID is already registered.'},
                        status=status.HTTP_409_CONFLICT)

    device = Device.objects.create(
        user=request.user, device_id=device_id, name=name, key=Device.new_key(),
    )
    return Response(_serialize(device, include_key=True), status=status.HTTP_201_CREATED)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_device(request, pk):
    """Delete a device owned by the caller (or any device for superusers)."""
    device = Device.objects.filter(pk=pk).first()
    if device is None:
        return Response({'error': 'Device not found.'}, status=status.HTTP_404_NOT_FOUND)
    if device.user != request.user and not request.user.is_superuser:
        return Response({'error': 'Permission denied.'}, status=status.HTTP_403_FORBIDDEN)
    device.delete()
    return Response({'message': 'Device removed.'}, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([AllowAny])
@throttle_classes([HeartbeatRateThrottle])
def heartbeat(request):
    """
    Device check-in. Authenticated by the per-device secret key sent in the
    ``X-Device-Key`` header (or ``key`` in the body). Updates last-seen so the
    UI can show online/offline status, plus optional battery/firmware/session.
    """
    key = request.headers.get('X-Device-Key') or request.data.get('key', '')
    if not key:
        return Response({'error': 'Device key required.'}, status=status.HTTP_401_UNAUTHORIZED)

    device = Device.objects.filter(key=key).first()
    if device is None:
        return Response({'error': 'Invalid device key.'}, status=status.HTTP_403_FORBIDDEN)

    device.last_seen = timezone.now()
    battery = request.data.get('battery')
    if battery is not None:
        try:
            device.battery = max(0, min(100, int(battery)))
        except (TypeError, ValueError):
            pass
    firmware = request.data.get('firmware')
    if firmware:
        device.firmware = str(firmware)[:50]
    session = request.data.get('session')
    if session is not None:
        device.current_session = str(session)[:200]
    device.save(update_fields=['last_seen', 'battery', 'firmware', 'current_session'])

    return Response({'status': 'ok', 'is_online': device.is_online})
