"""
PDF Health Report Generator
============================
Generates a one-page-per-signal PDF report for a session using reportlab.

GET /api/device/sessions/<owner>/<name>/report.pdf
"""
import io
import os
import datetime

from django.conf import settings
from django.http import FileResponse

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from login.views.device import (
    _session_dir, _detect_signals, _load_standard, _load_acc, _load_ibi,
    _read_info, SIGNAL_META,
)
from login.models import SessionAnnotation
import numpy as np


def _check_access(request, owner):
    if not request.user.is_superuser and request.user.username != owner:
        return Response({'error': 'Access denied.'}, status=403)
    return None


def _signal_stats(owner, name, session_path):
    """Compute a summary dict for each available signal."""
    results = {}

    for key in ('HR', 'EDA', 'TEMP'):
        meta = SIGNAL_META[key]
        p = os.path.join(session_path, meta['file'])
        if not os.path.isfile(p):
            continue
        r = _load_standard(p)
        if r is None:
            continue
        _, _, vals = r
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        results[key] = {
            'label': meta['label'], 'unit': meta['unit'],
            'min': float(np.min(vals)), 'max': float(np.max(vals)),
            'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
        }

    acc_path = os.path.join(session_path, 'ACC.csv')
    if os.path.isfile(acc_path):
        r = _load_acc(acc_path)
        if r is not None:
            _, _, mag = r
            mag = mag[np.isfinite(mag)]
            if len(mag):
                results['ACC'] = {
                    'label': 'Movement (ACC mag.)', 'unit': 'g',
                    'min': float(np.min(mag)), 'max': float(np.max(mag)),
                    'mean': float(np.mean(mag)), 'std': float(np.std(mag)),
                }

    ibi_path = os.path.join(session_path, 'IBI.csv')
    if os.path.isfile(ibi_path):
        r = _load_ibi(ibi_path)
        if r is not None:
            _, ibi_ms = r
            ibi_ms = ibi_ms[np.isfinite(ibi_ms)]
            if len(ibi_ms) >= 2:
                diffs = np.diff(ibi_ms)
                rmssd = float(np.sqrt(np.mean(diffs ** 2)))
                sdnn = float(np.std(ibi_ms))
                results['HRV'] = {
                    'label': 'Heart Rate Variability', 'unit': '',
                    'RMSSD': round(rmssd, 2), 'SDNN': round(sdnn, 2),
                    'mean_ibi': round(float(np.mean(ibi_ms)), 2),
                }

    return results


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def session_report_pdf(request, owner, name):
    err = _check_access(request, owner)
    if err:
        return err

    session_path = _session_dir(owner, name)
    if session_path is None:
        return Response({'error': 'Session not found.'}, status=404)

    stats = _signal_stats(owner, name, session_path)
    annotations = list(SessionAnnotation.objects.filter(owner=owner, session=name))
    info_text = _read_info(session_path)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    emerald = colors.HexColor('#10b981')
    dark = colors.HexColor('#111827')

    title_style = ParagraphStyle('title', parent=styles['Title'],
                                  fontSize=22, textColor=dark, spaceAfter=4)
    subtitle_style = ParagraphStyle('subtitle', parent=styles['Normal'],
                                     fontSize=10, textColor=colors.HexColor('#6b7280'), spaceAfter=16)
    heading_style = ParagraphStyle('heading', parent=styles['Heading2'],
                                    fontSize=13, textColor=emerald, spaceBefore=18, spaceAfter=6)
    body_style = ParagraphStyle('body', parent=styles['Normal'], fontSize=9,
                                 textColor=dark, leading=14)

    story = []

    # Header
    story.append(Paragraph('M-Health Signal Report', title_style))
    generated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    story.append(Paragraph(f'Session: <b>{name}</b> &nbsp;·&nbsp; Patient: <b>{owner}</b> &nbsp;·&nbsp; Generated: {generated}', subtitle_style))
    story.append(HRFlowable(width='100%', thickness=1, color=emerald, spaceAfter=14))

    # Signal summary table
    if stats:
        story.append(Paragraph('Signal Summary', heading_style))
        header = ['Signal', 'Unit', 'Min', 'Mean', 'Max', 'Std Dev']
        rows = [header]
        for key, s in stats.items():
            if key == 'HRV':
                rows.append(['HRV (IBI)', 'ms',
                              f"RMSSD {s['RMSSD']}", f"Mean IBI {s['mean_ibi']}",
                              f"SDNN {s['SDNN']}", '—'])
            else:
                rows.append([
                    s['label'], s['unit'],
                    f"{s['min']:.2f}", f"{s['mean']:.2f}",
                    f"{s['max']:.2f}", f"{s['std']:.2f}",
                ])

        tbl = Table(rows, colWidths=[4.5*cm, 1.8*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), emerald),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0fdf4'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#d1fae5')),
            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)

    # Annotations
    if annotations:
        story.append(Paragraph('Session Annotations', heading_style))
        ann_rows = [['Time (s)', 'Note']]
        for a in annotations:
            ann_rows.append([f"{a.offset_sec:.1f}s", a.text])
        ann_tbl = Table(ann_rows, colWidths=[3*cm, 12.1*cm])
        ann_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), emerald),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0fdf4'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#d1fae5')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(ann_tbl)

    # Session info
    if info_text:
        story.append(Paragraph('Session Info', heading_style))
        safe = info_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph(f'<font name="Courier" size="8">{safe}</font>', body_style))

    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#d1d5db')))
    story.append(Paragraph(
        '<font color="#9ca3af" size="8">This report is generated automatically from wearable device data. '
        'It is not a substitute for clinical judgement.</font>',
        ParagraphStyle('footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, spaceBefore=6),
    ))

    doc.build(story)
    buf.seek(0)
    return FileResponse(buf, as_attachment=True,
                        filename=f'{owner}_{name}_report.pdf',
                        content_type='application/pdf')
