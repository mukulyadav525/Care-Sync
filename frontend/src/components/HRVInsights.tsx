"use client";

import { useEffect, useRef, useState } from 'react';
import { HeartPulse, AlertTriangle, ShieldCheck, TrendingUp, Info, BellRing, Moon, Footprints, Armchair, Flame } from 'lucide-react';
import {
  ComposedChart, Area, Line as RLine, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import api from '@/lib/api';

const AI_API_URL = process.env.NEXT_PUBLIC_AI_API_URL || 'http://127.0.0.1:8001';
const FETCH_TIMEOUT_MS = 8000; // don't let a dead ai/ service hang the page

interface SeriesPoint { t: string; value: number; }
interface SignalsMap {
  HR?: { series: SeriesPoint[] };
  IBI?: { series: SeriesPoint[] };
  TEMP?: { series: SeriesPoint[] };
  EDA?: { series: SeriesPoint[] };
  ACC?: { series: SeriesPoint[] };
}

interface HorizonForecast {
  horizon_s: number;
  hr_pred: number;
  hr_lower: number;
  hr_upper: number;
  rmssd_pred: number | null;
  temp_pred: number | null;
  eda_pred: number | null;
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
  // ACC magnitude (g) — required for physiological-state classification
  // (sleep/rest/walking/exercise), which the digital twin and circadian
  // baseline depend on. Without this, state stays "unknown" forever and
  // resting/sleep/walking HR never populate.
  const accByT = new Map((signals.ACC?.series || []).map((p) => [p.t, p.value]));

  return hrSeries.map((p) => {
    const ibiMs = ibiByT.get(p.t);
    const sample: any = { timestamp: p.t, hr: p.value };
    if (ibiMs !== undefined) sample.ibi = ibiMs / 1000; // ms -> s
    const temp = tempByT.get(p.t);
    if (temp !== undefined) sample.temp = temp;
    const eda = edaByT.get(p.t);
    if (eda !== undefined) sample.eda = eda;
    const acc = accByT.get(p.t);
    if (acc !== undefined) sample.acc_mag = acc;
    return sample;
  });
}

function fetchWithTimeout(url: string, init: RequestInit, ms = FETCH_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return fetch(url, { ...init, signal: controller.signal }).finally(() => clearTimeout(timer));
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

// Short, unobtrusive "buzz" tone for serious alerts — no audio asset needed.
function playAlertTone() {
  try {
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.15, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.35);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.4);
  } catch { /* best-effort only */ }
}

function notifyAlert(title: string, body: string) {
  if (typeof window === 'undefined' || !('Notification' in window)) return;
  if (Notification.permission === 'granted') {
    new Notification(title, { body });
  } else if (Notification.permission !== 'denied') {
    Notification.requestPermission().then((perm) => {
      if (perm === 'granted') new Notification(title, { body });
    });
  }
}

const SEVERITY_COLOR: Record<string, string> = { normal: 'var(--success)', watch: '#f59e0b', alert: 'var(--error)' };

export default function HRVInsights({ signals, owner, session }: { signals: SignalsMap; owner: string; session: string }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [anomaly, setAnomaly] = useState<AnomalyResponse | null>(null);
  const [twin, setTwin] = useState<DigitalTwinResponse | null>(null);
  const reportedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const samples = buildSamples(signals);
    if (!samples) return;

    let cancelled = false;
    setLoading(true);
    setError('');

    const subjectId = `${owner}/${session}`;
    const body = JSON.stringify({ subject_id: subjectId, samples });
    const headers = { 'Content-Type': 'application/json' };

    Promise.allSettled([
      fetchWithTimeout(`${AI_API_URL}/hrv/forecast`, { method: 'POST', headers, body: JSON.stringify({ subject_id: subjectId, samples, horizons_s: [60, 300, 600] }) }).then((r) => r.json()),
      fetchWithTimeout(`${AI_API_URL}/hrv/anomaly`, { method: 'POST', headers, body }).then((r) => r.json()),
      fetchWithTimeout(`${AI_API_URL}/hrv/digital-twin`, { method: 'POST', headers, body }).then((r) => r.json()),
    ]).then(([f, a, t]) => {
      if (cancelled) return;
      if (f.status === 'fulfilled') setForecast(f.value); else setError((e) => e || 'Forecast unavailable');
      if (a.status === 'fulfilled') {
        const anomalyResult: AnomalyResponse = a.value;
        setAnomaly(anomalyResult);

        // Report non-normal severity to the backend (email on 'alert', dedup
        // handled server-side too, but skip the call entirely if we already
        // reported this exact owner/session/severity in this page view).
        if (anomalyResult.severity !== 'normal') {
          const key = `${owner}/${session}/${anomalyResult.severity}`;
          if (!reportedRef.current.has(key)) {
            reportedRef.current.add(key);
            api.post('/alerts/hrv/', {
              owner, session, severity: anomalyResult.severity, score: anomalyResult.score,
              reasons: anomalyResult.reasons, model_status: anomalyResult.model_status,
            }).catch(() => { /* best-effort — don't block the UI on this */ });
          }
          if (anomalyResult.severity === 'alert') {
            playAlertTone();
            notifyAlert('Care-Sync alert', `${owner}/${session}: ${anomalyResult.reasons[0] || 'unusual vitals detected'}`);
          }
        }
      } else {
        setError((e) => e || 'Anomaly check unavailable');
      }
      if (t.status === 'fulfilled') setTwin(t.value); else setError((e) => e || 'Digital twin unavailable');
    }).catch(() => { if (!cancelled) setError('HRV service unavailable'); })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [signals, owner, session]);

  const samples = buildSamples(signals);
  if (!samples) return null;

  const isAlert = anomaly?.severity === 'alert';
  const lastHr = samples[samples.length - 1]?.hr as number | undefined;

  const chartData = forecast ? [
    ...(lastHr != null ? [{ label: 'now', hr: lastHr, band: [lastHr, lastHr] as [number, number] }] : []),
    ...forecast.horizons.map((h) => ({
      label: `+${Math.round(h.horizon_s / 60)}m`,
      hr: h.hr_pred,
      band: [h.hr_lower, h.hr_upper] as [number, number],
    })),
  ] : [];

  const twinTiles = twin ? [
    { label: 'Resting', value: twin.resting_hr, icon: Armchair, color: '#3b82f6' },
    { label: 'Sleep', value: twin.sleep_hr, icon: Moon, color: '#6366f1' },
    { label: 'Walking', value: twin.walking_hr, icon: Footprints, color: '#10b981' },
    { label: 'Running/Exercise', value: twin.running_hr, icon: Flame, color: '#f59e0b' },
  ] : [];

  return (
    <div
      className="glass-panel"
      style={{
        padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem',
        border: isAlert ? '1px solid var(--error)' : undefined,
        animation: isAlert ? 'hrv-alert-pulse 1.6s ease-in-out infinite' : undefined,
      }}
    >
      <style>{`
        @keyframes hrv-alert-pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.25); }
          50% { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
        }
      `}</style>

      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.05rem' }}>
        {isAlert ? <BellRing size={20} color="var(--error)" /> : <HeartPulse size={20} color="var(--primary)" />} HRV Insights
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
                  {anomaly.severity === 'normal' ? 'No anomaly' : anomaly.severity === 'watch' ? 'Worth watching' : 'Alert — email sent'}
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
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                  Not fully calibrated — resting/sleep/walking HR need enough
                  movement (ACC) data to tell states apart, and enough
                  history for the circadian baseline. Improves automatically
                  the more this person is monitored.
                </p>
              )}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem' }}>
                {twinTiles.map((tile) => (
                  <div key={tile.label} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem', borderRadius: 'var(--radius-sm)', background: 'var(--panel-bg-light)' }}>
                    <tile.icon size={15} color={tile.value != null ? tile.color : 'var(--text-muted)'} />
                    <div>
                      <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{tile.label}</p>
                      <p style={{ fontSize: '0.85rem', fontWeight: 700 }}>{tile.value != null ? `${tile.value.toFixed(0)} bpm` : '—'}</p>
                    </div>
                  </div>
                ))}
              </div>
              <p style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
                Avg RMSSD: <strong>{twin.avg_rmssd != null ? `${twin.avg_rmssd.toFixed(1)} ms` : '—'}</strong>
              </p>
            </div>
          )}

          {/* Forecast (summary tiles — chart is below, full width) */}
          {forecast && (
            <div style={{ padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 700, fontSize: '0.9rem' }}>
                  <TrendingUp size={16} /> Vitals Forecast
                </span>
                {statusBadge(forecast.model_status)}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontSize: '0.82rem' }}>
                {forecast.horizons.map((h) => (
                  <span key={h.horizon_s}>
                    +{Math.round(h.horizon_s / 60)} min: <strong>{h.hr_pred.toFixed(0)} bpm</strong>
                    {h.rmssd_pred != null && <span style={{ color: 'var(--text-muted)' }}> · RMSSD {h.rmssd_pred.toFixed(0)}ms</span>}
                    {h.temp_pred != null && <span style={{ color: 'var(--text-muted)' }}> · {h.temp_pred.toFixed(1)}°C</span>}
                    {h.eda_pred != null && <span style={{ color: 'var(--text-muted)' }}> · {h.eda_pred.toFixed(2)}µS</span>}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* HR forecast chart — trajectory + confidence band */}
      {forecast && chartData.length > 1 && (
        <div>
          <p style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            HR trajectory (shaded band = forecast uncertainty)
          </p>
          <div style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} unit=" bpm" />
                <Tooltip formatter={((v: any, name: any) => (name === 'band' ? ['', ''] : [`${Math.round(v)} bpm`, 'HR'])) as any} />
                <Area dataKey="band" stroke="none" fill="var(--primary)" fillOpacity={0.12} isAnimationActive={false} />
                <RLine type="monotone" dataKey="hr" stroke="var(--primary)" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
