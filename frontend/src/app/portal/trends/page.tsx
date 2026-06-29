"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';
import { TrendingUp, ArrowLeft, HeartPulse, Droplets, Thermometer, Activity, Zap } from 'lucide-react';
import api from '@/lib/api';

interface TrendPoint {
  name: string;
  start: string | null;
  HR?: number;
  EDA?: number;
  TEMP?: number;
  ACC?: number;
  RMSSD?: number;
}

const METRICS: { key: keyof Omit<TrendPoint, 'name' | 'start'>; label: string; unit: string; color: string; icon: any }[] = [
  { key: 'HR',    label: 'Avg Heart Rate',     unit: 'bpm', color: '#ef4444', icon: HeartPulse },
  { key: 'EDA',   label: 'Avg Skin Conduct.',  unit: 'µS',  color: '#0d9488', icon: Droplets },
  { key: 'TEMP',  label: 'Avg Temperature',    unit: '°C',  color: '#f59e0b', icon: Thermometer },
  { key: 'ACC',   label: 'Avg Movement',       unit: 'g',   color: '#10b981', icon: Activity },
  { key: 'RMSSD', label: 'HRV (RMSSD)',        unit: 'ms',  color: '#ec4899', icon: Zap },
];

function fmtDate(iso: string | null) {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export default function TrendsPage() {
  const router = useRouter();
  const [data, setData] = useState<TrendPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeKeys, setActiveKeys] = useState<Set<string>>(new Set(['HR', 'EDA', 'TEMP']));

  useEffect(() => {
    api.get('/device/sessions/trends/').then((r) => {
      setData(r.data.trends || []);
      setLoading(false);
    }).catch(() => router.push('/login'));
  }, [router]);

  const toggleKey = (key: string) => {
    setActiveKeys((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  if (loading) return <div className="container" style={{ padding: '4rem', textAlign: 'center' }}>Loading trends…</div>;

  const availableMetrics = METRICS.filter((m) => data.some((d) => d[m.key] !== undefined));

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <Link href="/portal" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
          <ArrowLeft size={15} /> Sessions
        </Link>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <TrendingUp size={24} color="var(--primary)" /> Signal Trends
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
          Average signal values across all your sessions over time.
        </p>
      </div>

      {data.length < 2 ? (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <TrendingUp size={40} style={{ marginBottom: '1rem', opacity: 0.25 }} />
          <p>Need at least 2 sessions to show trends. Record more sessions first.</p>
        </div>
      ) : (
        <>
          {/* Metric toggles */}
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {availableMetrics.map((m) => (
              <button key={m.key} onClick={() => toggleKey(m.key)}
                style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.4rem 0.85rem', border: `1px solid ${activeKeys.has(m.key) ? m.color : 'var(--border)'}`, borderRadius: '999px', background: activeKeys.has(m.key) ? `${m.color}15` : 'transparent', color: activeKeys.has(m.key) ? m.color : 'var(--text-muted)', cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600, transition: 'all 0.15s' }}>
                <m.icon size={13} /> {m.label}
              </button>
            ))}
          </div>

          {/* Main trend chart */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h3 style={{ marginBottom: '1.25rem', fontSize: '1rem' }}>All signals over sessions</h3>
            <div style={{ width: '100%', height: 320 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 5, right: 20, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="start" tickFormatter={fmtDate} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={{ stroke: 'var(--border)' }} minTickGap={40} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={{ stroke: 'var(--border)' }} width={40} />
                  <Tooltip
                    labelFormatter={(iso) => `Session: ${fmtDate(iso as string)}`}
                    contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '8px', fontSize: '0.8rem' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '0.8rem', paddingTop: '1rem' }} />
                  {availableMetrics.filter((m) => activeKeys.has(m.key)).map((m) => (
                    <Line key={m.key} type="monotone" dataKey={m.key} name={m.label} stroke={m.color} strokeWidth={2} dot={{ r: 4, fill: m.color }} connectNulls />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Per-metric mini cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
            {availableMetrics.map((m) => {
              const vals = data.map((d) => d[m.key]).filter((v): v is number => v !== undefined);
              if (!vals.length) return null;
              const first = vals[0], last = vals[vals.length - 1];
              const delta = last - first;
              const pct = first !== 0 ? (delta / Math.abs(first)) * 100 : 0;
              const trend = Math.abs(pct) < 1 ? 'stable' : pct > 0 ? 'up' : 'down';
              const trendColor = trend === 'stable' ? 'var(--text-muted)' : m.key === 'HR' ? (trend === 'down' ? '#10b981' : '#ef4444') : '#10b981';
              return (
                <div key={m.key} className="glass-panel" style={{ padding: '1.1rem', display: 'flex', gap: '0.85rem', alignItems: 'center' }}>
                  <div style={{ background: `${m.color}15`, padding: '0.75rem', borderRadius: '50%', display: 'flex', flexShrink: 0 }}>
                    <m.icon size={20} color={m.color} />
                  </div>
                  <div>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{m.label}</p>
                    <p style={{ fontWeight: 700, fontSize: '1.1rem' }}>{last.toFixed(1)} <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{m.unit}</span></p>
                    <p style={{ fontSize: '0.75rem', color: trendColor, fontWeight: 600 }}>
                      {trend === 'stable' ? '~ stable' : `${pct > 0 ? '↑' : '↓'} ${Math.abs(pct).toFixed(1)}% vs first`}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Session data table */}
          <div className="glass-panel" style={{ overflow: 'hidden' }}>
            <h3 style={{ padding: '1.25rem 1.5rem 0', fontSize: '1rem', marginBottom: '0.75rem' }}>Session averages</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.88rem' }}>
                <thead>
                  <tr style={{ background: 'var(--panel-bg-light)', borderBottom: '1px solid var(--border)' }}>
                    <th style={{ padding: '0.75rem 1.5rem' }}>Session</th>
                    <th style={{ padding: '0.75rem 1rem' }}>Date</th>
                    {availableMetrics.map((m) => <th key={m.key} style={{ padding: '0.75rem 1rem' }}>{m.key} ({m.unit})</th>)}
                  </tr>
                </thead>
                <tbody>
                  {data.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '0.75rem 1.5rem' }}>
                        <Link href={`/portal/${encodeURIComponent(row.name.split('/')[0] || '')}/${encodeURIComponent(row.name)}`} style={{ fontWeight: 600, color: 'var(--primary)' }}>
                          {row.name}
                        </Link>
                      </td>
                      <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>{fmtDate(row.start)}</td>
                      {availableMetrics.map((m) => (
                        <td key={m.key} style={{ padding: '0.75rem 1rem', color: row[m.key] !== undefined ? 'inherit' : 'var(--text-muted)' }}>
                          {row[m.key]?.toFixed(1) ?? '—'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
