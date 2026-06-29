"""
Alert Rules API
================
Users define threshold rules (e.g. avg HR > 110). When a session is fetched
via the portal, evaluate_alerts() checks the session stats and returns fired
alerts alongside the data.

Endpoints:
  GET  /api/alerts/           list rules for the authenticated user
  POST /api/alerts/           create a rule
  PATCH /api/alerts/<pk>/     update (enable/disable/change threshold)
  DELETE /api/alerts/<pk>/    delete
"""
from rest_framework import status as drf_status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from login.models import AlertRule, AlertFired


VALID_SIGNALS = {c[0] for c in AlertRule.SIGNAL_CHOICES}
VALID_OPS = {c[0] for c in AlertRule.OP_CHOICES}


def _rule_json(r):
    return {
        'id': r.id, 'signal': r.signal, 'operator': r.operator,
        'threshold': r.threshold, 'label': r.label, 'enabled': r.enabled,
        'created_at': r.created_at.isoformat(),
    }


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def alert_list(request):
    if request.method == 'GET':
        rules = AlertRule.objects.filter(user=request.user)
        return Response({'rules': [_rule_json(r) for r in rules]})

    # POST — create
    signal = request.data.get('signal', '')
    operator = request.data.get('operator', '')
    label = str(request.data.get('label', '')).strip()

    if signal not in VALID_SIGNALS:
        return Response({'error': f'signal must be one of {list(VALID_SIGNALS)}'}, status=400)
    if operator not in VALID_OPS:
        return Response({'error': 'operator must be "gt" or "lt"'}, status=400)
    try:
        threshold = float(request.data.get('threshold'))
    except (TypeError, ValueError):
        return Response({'error': 'threshold must be a number'}, status=400)

    r = AlertRule.objects.create(
        user=request.user, signal=signal, operator=operator,
        threshold=threshold, label=label,
    )
    return Response(_rule_json(r), status=201)


@api_view(['PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def alert_detail(request, pk):
    try:
        rule = AlertRule.objects.get(pk=pk, user=request.user)
    except AlertRule.DoesNotExist:
        return Response({'error': 'Not found.'}, status=404)

    if request.method == 'DELETE':
        rule.delete()
        return Response(status=204)

    # PATCH
    if 'threshold' in request.data:
        try:
            rule.threshold = float(request.data['threshold'])
        except (TypeError, ValueError):
            return Response({'error': 'threshold must be a number'}, status=400)
    if 'label' in request.data:
        rule.label = str(request.data['label']).strip()
    if 'enabled' in request.data:
        rule.enabled = bool(request.data['enabled'])
    if 'operator' in request.data:
        op = request.data['operator']
        if op not in VALID_OPS:
            return Response({'error': 'Invalid operator'}, status=400)
        rule.operator = op
    rule.save()
    return Response(_rule_json(rule))


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def alert_history(request):
    """GET /api/alerts/history/ — last 50 fired alerts for the user."""
    qs = AlertFired.objects.filter(user=request.user).select_related('rule')[:50]
    return Response({'history': [
        {
            'id': f.id,
            'label': f.label,
            'signal': f.signal,
            'operator': f.operator,
            'threshold': f.threshold,
            'actual_mean': f.actual_mean,
            'owner': f.owner,
            'session': f.session,
            'fired_at': f.fired_at.isoformat(),
        }
        for f in qs
    ]})


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def clear_alert_history(request):
    """DELETE /api/alerts/history/ — clear all history for the user."""
    AlertFired.objects.filter(user=request.user).delete()
    return Response(status=204)


def evaluate_alerts(user, signal_stats: dict) -> list[dict]:
    """
    Evaluate all enabled rules for this user against a dict of signal stats.
    signal_stats: { 'HR': {'mean': 72, ...}, 'EDA': {...}, ... }
    Returns a list of fired alert dicts.
    """
    fired = []
    rules = AlertRule.objects.filter(user=user, enabled=True)
    for rule in rules:
        sig_data = signal_stats.get(rule.signal)
        if sig_data is None:
            continue
        value = sig_data.get('mean')
        if value is None:
            continue
        triggered = (rule.operator == 'gt' and value > rule.threshold) or \
                    (rule.operator == 'lt' and value < rule.threshold)
        if triggered:
            op_str = '>' if rule.operator == 'gt' else '<'
            label = rule.label or f'{rule.signal} {op_str} {rule.threshold}'
            fired.append({
                'rule_id': rule.id,
                'signal': rule.signal,
                'label': label,
                'operator': rule.operator,
                'threshold': rule.threshold,
                'actual_mean': round(value, 2),
            })
    return fired


def persist_fired_alerts(user, fired: list[dict], owner: str, session: str):
    """Save fired alerts to AlertFired history, skipping duplicates for same session."""
    existing = set(
        AlertFired.objects.filter(user=user, owner=owner, session=session)
        .values_list('signal', flat=True)
    )
    to_create = [
        AlertFired(
            user=user,
            rule_id=f.get('rule_id'),
            signal=f['signal'],
            label=f['label'],
            operator=f['operator'],
            threshold=f['threshold'],
            actual_mean=f['actual_mean'],
            owner=owner,
            session=session,
        )
        for f in fired if f['signal'] not in existing
    ]
    if to_create:
        AlertFired.objects.bulk_create(to_create)
