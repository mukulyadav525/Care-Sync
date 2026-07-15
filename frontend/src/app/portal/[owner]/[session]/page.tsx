"use client";

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import {
  ArrowLeft, Clock, Tag, HeartPulse, Activity, Thermometer, Droplets, Waves, Zap,
  FileText, AlertTriangle, Plus, Trash2, GitCompare,
} from 'lucide-react';
import api, { downloadFile } from '@/lib/api';
import HRVInsights from '@/components/HRVInsights';

type Granularity = 'minute' | 'hour' | 'day';

interface SignalData {
  key: string;
  label: string;
  unit: string;
  color: string;
  mode: 'continuous' | 'waveform' | 'event';
  sample_rate?: number;
  window_sec?: number;
  series: { t: string; value: number }[];
  stats: Record<string, number>;
}

interface Annotation {
  id: number;
  offset_sec: number;
  text: string;
  created_at: string;
}

interface FiredAlert {
  rule_id: number;
  signal: string;
  label: string;
  operator: string;
  threshold: number;
  actual_mean: number;
}

interface SessionDetail {
  owner: string;
  name: string;
  granularity: Granularity;
  start: string | null;
  end: string | null;
  duration_sec: number;
  info: string | null;
  tags: string[];
  signals: Record<string, SignalData>;
  daily: Record<string, any>[];
  alerts: FiredAlert[];
}

const SIGNAL_ICONS: Record<string, any> = {
  HR: HeartPulse, EDA: Droplets, TEMP: Thermometer, ACC: Activity, BVP: Waves, IBI: Zap,
};
const SIGNAL_ORDER = ['HR', 'EDA', 'TEMP', 'ACC', 'IBI', 'BVP'];

function fmtDuration(sec: number): string {
  if (!sec) return '—';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

export default function SessionDashboard() {
  const params = useParams();
  const owner = decodeURIComponent(params.owner as string);
  const session = decodeURIComponent(params.session as string);

  const [granularity, setGranularity] = useState<Granularity>('hour');
  const [data, setData] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [newNote, setNewNote] = useState('');
  const [newOffset, setNewOffset] = useState('0');
  const [addingNote, setAddingNote] = useState(false);

  const [pdfLoading, setPdfLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const fetchData = async () => {
      setLoading(true);
      setError('');
      try {
        const [res, annRes] = await Promise.all([
          api.get(`/device/sessions/${encodeURIComponent(owner)}/${encodeURIComponent(session)}/?granularity=${granularity}`),
          api.get(`/device/sessions/${encodeURIComponent(owner)}/${encodeURIComponent(session)}/annotations/`),
        ]);
        if (!cancelled) {
          setData(res.data);
          setAnnotations(annRes.data.annotations || []);
        }
      } catch (err: any) {
        if (!cancelled) setError(err.response?.data?.error || 'Failed to load session.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchData();
    return () => { cancelled = true; };
  }, [owner, session, granularity]);

  const fmtTick = (iso: string) => {
    const d = new Date(iso);
    if (granularity === 'day') return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  };
  const fmtTooltipLabel = (iso: string) =>
    new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });

  const handleAddAnnotation = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newNote.trim()) return;
    setAddingNote(true);
    try {
      const res = await api.post(
        `/device/sessions/${encodeURIComponent(owner)}/${encodeURIComponent(session)}/annotations/`,
        { offset_sec: parseFloat(newOffset) || 0, text: newNote.trim() },
      );
      setAnnotations((prev) => [...prev, res.data].sort((a, b) => a.offset_sec - b.offset_sec));
      setNewNote('');
      setNewOffset('0');
    } catch { /* ignore */ }
    finally { setAddingNote(false); }
  };

  const handleDeleteAnnotation = async (id: number) => {
    await api.delete(`/device/annotations/${id}/`);
    setAnnotations((prev) => prev.filter((a) => a.id !== id));
  };

  const handleDownloadPdf = async () => {
    setPdfLoading(true);
    try {
      await downloadFile(
        `/device/sessions/${encodeURIComponent(owner)}/${encodeURIComponent(session)}/report.pdf`,
        `${owner}_${session}_report.pdf`,
      );
    } catch { /* ignore */ }
    finally { setPdfLoading(false); }
  };

  if (loading && !data) {
    return <div className="container" style={{ padding: '4rem', textAlign: 'center' }}>Loading signals…</div>;
  }
  if (error) {
    return (
      <div className="container" style={{ padding: '4rem', textAlign: 'center' }}>
        <p style={{ color: 'var(--error)', marginBottom: '1rem' }}>{error}</p>
        <Link href="/portal" className="btn btn-outline">← Back to sessions</Link>
      </div>
    );
  }
  if (!data) return null;

  const signals = SIGNAL_ORDER.map((k) => data.signals[k]).filter(Boolean) as SignalData[];

  const summary = [
    { label: 'Avg Heart Rate', value: data.signals.HR?.stats.avg, unit: 'bpm', icon: HeartPulse, color: '#ef4444' },
    { label: 'HRV (RMSSD)', value: data.signals.IBI?.stats.rmssd, unit: 'ms', icon: Zap, color: '#ec4899' },
    { label: 'Avg Skin Conductance', value: data.signals.EDA?.stats.avg, unit: 'µS', icon: Droplets, color: '#0d9488' },
    { label: 'Avg Temperature', value: data.signals.TEMP?.stats.avg, unit: '°C', icon: Thermometer, color: '#f59e0b' },
  ].filter((m) => m.value !== undefined);

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <Link href="/portal" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
            <ArrowLeft size={15} /> Sessions
          </Link>
          <h2>{session}</h2>
          <div style={{ display: 'flex', gap: '1.25rem', flexWrap: 'wrap', color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.4rem' }}>
            <span>{owner}</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}><Clock size={14} /> {fmtDuration(data.duration_sec)}</span>
            {data.tags.length > 0 && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}><Tag size={14} /> {data.tags.length} events</span>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
          {/* Compare link */}
          <Link href={`/portal/compare?a=${encodeURIComponent(owner + '/' + session)}`} className="btn btn-outline" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}>
            <GitCompare size={16} /> Compare
          </Link>
          {/* PDF download */}
          <button onClick={handleDownloadPdf} className="btn btn-outline" disabled={pdfLoading}
            style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem' }}>
            <FileText size={16} /> {pdfLoading ? 'Generating…' : 'Export PDF'}
          </button>
          {/* Granularity */}
          <div style={{ display: 'flex', backgroundColor: 'var(--btn-toggle-bg)', borderRadius: 'var(--radius-md)', padding: '0.25rem' }}>
            {(['minute', 'hour', 'day'] as Granularity[]).map((g) => (
              <button key={g} onClick={() => setGranularity(g)} style={{
                padding: '0.5rem 1rem', borderRadius: 'calc(var(--radius-md) - 0.25rem)',
                border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem',
                textTransform: 'capitalize', transition: 'var(--transition-fast)',
                backgroundColor: granularity === g ? 'var(--btn-toggle-active)' : 'transparent',
                color: granularity === g ? 'var(--btn-toggle-text-active)' : 'var(--btn-toggle-text)',
              }}>
                {g === 'minute' ? 'Per Minute' : g === 'hour' ? 'Hourly' : 'Daily'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Fired alerts */}
      {data.alerts.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {data.alerts.map((a) => (
            <div key={a.rule_id} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.75rem 1rem', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', borderRadius: 'var(--radius-md)', color: '#ef4444', fontSize: '0.85rem' }}>
              <AlertTriangle size={16} />
              <strong>{a.label}</strong>
              <span style={{ color: 'var(--text-muted)' }}>avg {a.signal} = {a.actual_mean} (threshold {a.operator === 'gt' ? '>' : '<'} {a.threshold})</span>
            </div>
          ))}
        </div>
      )}

      {/* Summary metrics */}
      {summary.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
          {summary.map((m) => (
            <div key={m.label} className="glass-panel" style={{ padding: '1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <div style={{ background: `${m.color}1a`, padding: '0.85rem', borderRadius: '50%', display: 'flex' }}>
                <m.icon color={m.color} size={22} />
              </div>
              <div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{m.label}</p>
                <h3 style={{ fontSize: '1.4rem' }}>{m.value}<span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 500 }}> {m.unit}</span></h3>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* HRV forecast / anomaly / digital twin (ai/ service) */}
      <HRVInsights signals={data.signals} owner={owner} session={session} />

      {/* Signal panels */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 480px), 1fr))', gap: '1.5rem' }}>
        {signals.map((sig) => {
          const Icon = SIGNAL_ICONS[sig.key] || Activity;
          const isHRV = sig.key === 'IBI';
          return (
            <div key={sig.key} className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.05rem' }}>
                  <Icon size={18} color={sig.color} /> {sig.label}
                </h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {sig.mode === 'waveform' ? `${sig.window_sec}s raw @ ${sig.sample_rate}Hz` :
                   sig.mode === 'event' ? `${sig.stats.count} beats` : sig.unit}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                {isHRV ? (
                  <>
                    <Stat label="RMSSD" value={sig.stats.rmssd} unit="ms" />
                    <Stat label="SDNN" value={sig.stats.sdnn} unit="ms" />
                    <Stat label="Mean HR" value={sig.stats.mean_hr} unit="bpm" />
                  </>
                ) : (
                  <>
                    <Stat label="Min" value={sig.stats.min} unit={sig.unit} />
                    <Stat label="Avg" value={sig.stats.avg} unit={sig.unit} />
                    <Stat label="Max" value={sig.stats.max} unit={sig.unit} />
                  </>
                )}
              </div>
              <div style={{ width: '100%', height: 220 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={sig.series} margin={{ top: 5, right: 12, left: -8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis
                      dataKey="t"
                      tickFormatter={sig.mode === 'waveform'
                        ? (iso) => new Date(iso).toLocaleTimeString(undefined, { minute: '2-digit', second: '2-digit' })
                        : fmtTick}
                      tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                      axisLine={{ stroke: 'var(--border)' }}
                      minTickGap={40}
                    />
                    <YAxis domain={['auto', 'auto']} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={{ stroke: 'var(--border)' }} width={44} />
                    <Tooltip
                      labelFormatter={(label) => sig.mode === 'waveform' ? String(label) : fmtTooltipLabel(String(label))}
                      contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '8px', fontSize: '0.8rem' }}
                      itemStyle={{ color: sig.color }}
                      formatter={(v: any) => [`${v} ${sig.unit}`, sig.label]}
                    />
                    <Line type="monotone" dataKey="value" stroke={sig.color} strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          );
        })}
      </div>

      {/* Daily breakdown */}
      {data.daily.length > 0 && (
        <div className="glass-panel" style={{ overflow: 'hidden' }}>
          <h3 style={{ padding: '1.25rem 1.5rem 0' }}>Day-by-day averages</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', marginTop: '1rem' }}>
            <thead>
              <tr style={{ background: 'var(--panel-bg-light)', borderBottom: '1px solid var(--border)' }}>
                <th style={{ padding: '0.75rem 1.5rem' }}>Date</th>
                {Object.keys(data.daily[0]).filter((k) => k !== 'date').map((k) => (
                  <th key={k} style={{ padding: '0.75rem 1rem' }}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.daily.map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '0.75rem 1.5rem', fontWeight: 600 }}>{row.date}</td>
                  {Object.keys(row).filter((k) => k !== 'date').map((k) => (
                    <td key={k} style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>{row[k] ?? '—'}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Annotations */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Tag size={18} color="var(--primary)" /> Session Notes
        </h3>
        {annotations.length === 0 && <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1rem' }}>No notes yet.</p>}
        {annotations.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1rem' }}>
            {annotations.map((a) => (
              <div key={a.id} style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', padding: '0.75rem', background: 'var(--info-bg)', border: '1px solid var(--info-border)', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--text-muted)', minWidth: 56, paddingTop: 2 }}>{a.offset_sec.toFixed(1)}s</span>
                <span style={{ flex: 1, fontSize: '0.9rem' }}>{a.text}</span>
                <button onClick={() => handleDeleteAnnotation(a.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 0, display: 'flex' }}>
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
        )}
        <form onSubmit={handleAddAnnotation} style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <input
            type="number" step="0.1" min="0" placeholder="Time (s)" value={newOffset}
            onChange={(e) => setNewOffset(e.target.value)}
            style={{ width: 100, padding: '0.6rem 0.75rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontSize: '0.9rem' }}
          />
          <input
            type="text" placeholder="Add a note…" value={newNote}
            onChange={(e) => setNewNote(e.target.value)} required
            style={{ flex: 1, minWidth: 200, padding: '0.6rem 0.75rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontSize: '0.9rem' }}
          />
          <button type="submit" className="btn btn-primary" disabled={addingNote} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', whiteSpace: 'nowrap' }}>
            <Plus size={16} /> Add Note
          </button>
        </form>
      </div>
    </div>
  );
}

function Stat({ label, value, unit }: { label: string; value?: number; unit: string }) {
  return (
    <div style={{ flex: 1, minWidth: 80, background: 'var(--info-bg)', border: '1px solid var(--info-border)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem' }}>
      <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</p>
      <p style={{ fontWeight: 700, fontSize: '1.05rem' }}>
        {value ?? '—'}<span style={{ fontSize: '0.7rem', fontWeight: 500, color: 'var(--text-muted)' }}> {unit}</span>
      </p>
    </div>
  );
}
