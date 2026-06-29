"""
Session Annotation API — timestamped notes on any signal session.

Endpoints:
  GET  /api/device/sessions/<owner>/<name>/annotations/   list annotations
  POST /api/device/sessions/<owner>/<name>/annotations/   create annotation
  DELETE /api/device/annotations/<pk>/                     delete one
"""
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from login.models import SessionAnnotation


def _check_session_access(request, owner):
    """Return error Response if the requesting user can't access this owner's data."""
    if not request.user.is_superuser and request.user.username != owner:
        return Response({'error': 'Access denied.'}, status=status.HTTP_403_FORBIDDEN)
    return None


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def annotation_list(request, owner, name):
    err = _check_session_access(request, owner)
    if err:
        return err

    if request.method == 'GET':
        qs = SessionAnnotation.objects.filter(owner=owner, session=name)
        return Response({'annotations': [
            {'id': a.id, 'offset_sec': a.offset_sec, 'text': a.text,
             'created_at': a.created_at.isoformat()}
            for a in qs
        ]})

    # POST — create
    text = str(request.data.get('text', '')).strip()
    try:
        offset = float(request.data.get('offset_sec', 0))
    except (TypeError, ValueError):
        return Response({'error': 'offset_sec must be a number.'}, status=400)
    if not text:
        return Response({'error': 'text is required.'}, status=400)

    a = SessionAnnotation.objects.create(
        user=request.user, owner=owner, session=name,
        offset_sec=offset, text=text,
    )
    return Response({'id': a.id, 'offset_sec': a.offset_sec, 'text': a.text,
                     'created_at': a.created_at.isoformat()}, status=201)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def annotation_delete(request, pk):
    try:
        a = SessionAnnotation.objects.get(pk=pk)
    except SessionAnnotation.DoesNotExist:
        return Response({'error': 'Not found.'}, status=404)
    if not request.user.is_superuser and a.user != request.user:
        return Response({'error': 'Access denied.'}, status=403)
    a.delete()
    return Response(status=204)
