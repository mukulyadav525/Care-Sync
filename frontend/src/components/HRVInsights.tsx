"use client";

import { useEffect, useState } from 'react';
import { HeartPulse, AlertTriangle, ShieldCheck, TrendingUp, Info } from 'lucide-react';

const AI_API_URL = process.env.NEXT_PUBLIC_AI_API_URL || 'http://127.0.0.1:8001';

interface SeriesPoint { t: string; value: number; }
interface SignalsMap {
  HR?: { series: SeriesPoint[] };
  IBI?: { series: SeriesPoint[] };
  TEMP?: { series: SeriesPoint[] };
  EDA?: { series: SeriesPoint[] };
}

interface HorizonForecast {
  horizon_s: number;
  hr_pred: number;
  hr_lower: number;
  hr_upper: number;
  rmssd_pred: number | null;
}
interface ForecastResponse { model_status: string; horizons: HorizonForecast[]; }
interface AnomalyResponse {
  model_status: string;
  is_anomaly: boolean;
  score: number;
  severity: 'normal' | 'watch' | 'alert';
  reasons: string[];
}
interface DigitalTwinResponse {
  model_status: string;
  calibrated: boolean;
  resting_hr: number | null;
  sleep_hr: number | null;
  walking_hr: number | null;
  running_hr: number | null;
  avg_rmssd: number | null;
}

function buildSamples(signals: SignalsMap) {
  const hrSeries = signals.HR?.series || [];
  if (hrSeries.length < 2) return null;

  const ibiByT = new Map((signals.IBI?.series || []).map((p) => [p.t, p.value]));
  const tempByT = new Map((signals.TEMP?.series || []).map((p) => [p.t, p.value]));
  const edaByT = new Map((signals.EDA?.series || []).map((p) => [p.t, p.value]));

  return hrSeries.map((p) => {
    const ibiMs = ibiByT.get(p.t);
    const sample: any = { timestamp: p.t, hr: p.value };
    if (ibiMs !== undefined) sample.ibi = ibiMs / 1000; // ms -> s
    const temp = tempByT.get(p.t);
    if (temp !== undefined) sample.temp = temp;
    const eda = edaByT.get(p.t);
    if (eda !== undefined) sample.eda = eda;
    return sample;
  });
}

function statusBadge(status: string) {
  const map: Record<string, { label: string; color: string }> = {
    trained: { label: 'ML model', color: '#10b981' },
    pipeline: { label: 'live analysis', color: '#3b82f6' },
    mock: { label: 'estimate', color: '#f59e0b' },
  };
  const s = map[status] || map.mock;
  return (
    <span style={{ fontSize: '0.68rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.03em', padding: '0.15rem 0.5rem', borderRadius: '999px', color: s.color, background: `${s.color}1a` }}>
      {s.label}
    </span>
  );
}

const SEVERITY_COLOR: Record<string, string> = { normal: 'var(--success)', watch: '#f59e0b', alert: 'var(--error)' };

export default function HRVInsights({ signals }: { signals: SignalsMap }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [anomaly, setAnomaly] = useState<AnomalyResponse | null>(null);
  const [twin, setTwin] = useState<DigitalTwinResponse | null>(null);

  useEffect(() => {
    const samples = buildSamples(signals);
    if (!samples) return;

    let cancelled = false;
    setLoading(true);
    setError('');

    const subjectId = 'session';
    const body = JSON.stringify({ subject_id: subjectId, samples });
    const headers = { 'Content-Type': 'application/json' };

    Promise.allSettled([
      fetch(`${AI_API_URL}/hrv/forecast`, { method: 'POST', headers, body: JSON.stringify({ subject_id: subjectId, samples, horizons_s: [60, 300, 600] }) }).then((r) => r.json()),
      fetch(`${AI_API_URL}/hrv/anomaly`, { method: 'POST', headers, body }).then((r) => r.json()),
      fetch(`${AI_API_URL}/hrv/digital-twin`, { method: 'POST', headers, body }).then((r) => r.json()),
    ]).then(([f, a, t]) => {
      if (cancelled) return;
      if (f.status === 'fulfilled') setForecast(f.value); else setError((e) => e || 'Forecast unavailable');
      if (a.status === 'fulfilled') setAnomaly(a.value); else setError((e) => e || 'Anomaly check unavailable');
      if (t.status === 'fulfilled') setTwin(t.value); else setError((e) => e || 'Digital twin unavailable');
    }).catch(() => { if (!cancelled) setError('HRV service unavailable'); })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [signals]);

  const samples = buildSamples(signals);
  if (!samples) return null;

  return (
    <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.05rem' }}>
        <HeartPulse size={20} color="var(--primary)" /> HRV Insights
      </h3>

      {loading && <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Analyzing…</p>}
      {error && !loading && (
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Info size={14} /> {error} — is the AI service running?
        </p>
      )}

      {!loading && (anomaly || twin || forecast) && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>

          {/* Anomaly */}
          {anomaly && (
            <div style={{ padding: '1rem', borderRadius: 'var(--radius-md)', border: `1px solid ${SEVERITY_COLOR[anomaly.severity]}40`, background: `${SEVERITY_COLOR[anomaly.severity]}0d` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 700, fontSize: '0.9rem', color: SEVERITY_COLOR[anomaly.severity] }}>
                  {anomaly.severity === 'normal' ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />}
                  {anomaly.severity === 'normal' ? 'No anomaly' : anomaly.severity === 'watch' ? 'Worth watching' : 'Alert'}
                </span>
                {statusBadge(anomaly.model_status)}
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>Score: {anomaly.score.toFixed(2)}</p>
              {anomaly.reasons.map((r, i) => (
                <p key={i} style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{r}</p>
              ))}
            </div>
          )}

          {/* Digital twin */}
          {twin && (
            <div style={{ padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>Digital Twin Baseline</span>
                {statusBadge(twin.model_status)}
              </div>
              {!twin.calibrated && (
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                  Not fully calibrated yet — more history improves accuracy.
                </p>
              )}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontSize: '0.82rem' }}>
                <span>Resting HR: <strong>{twin.resting_hr != null ? `${twin.resting_hr.toFixed(0)} bpm` : '—'}</strong></span>
                <span>Sleep HR: <strong>{twin.sleep_hr != null ? `${twin.sleep_hr.toFixed(0)} bpm` : '—'}</strong></span>
                <span>Walking HR: <strong>{twin.walking_hr != null ? `${twin.walking_hr.toFixed(0)} bpm` : '—'}</strong></span>
                <span>Avg RMSSD: <strong>{twin.avg_rmssd != null ? `${twin.avg_rmssd.toFixed(1)} ms` : '—'}</strong></span>
              </div>
            </div>
          )}

          {/* Forecast */}
          {forecast && (
            <div style={{ padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 700, fontSize: '0.9rem' }}>
                  <TrendingUp size={16} /> HR Forecast
                </span>
                {statusBadge(forecast.model_status)}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontSize: '0.82rem' }}>
                {forecast.horizons.map((h) => (
                  <span key={h.horizon_s}>
                    +{Math.round(h.horizon_s / 60)} min: <strong>{h.hr_pred.toFixed(0)} bpm</strong>
                    <span style={{ color: 'var(--text-muted)' }}> ({h.hr_lower.toFixed(0)}–{h.hr_upper.toFixed(0)})</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
