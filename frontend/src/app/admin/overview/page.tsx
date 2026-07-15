"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  LayoutDashboard, Users, Radio, FileText, Bell, HeartPulse, ChevronRight, Mail, TrendingUp,
} from 'lucide-react';
import api from '@/lib/api';

interface Overview {
  users: { total: number; with_sessions: number; with_consent: number };
  sessions: { total: number };
  devices: { total: number; online: number };
  documents_by_type: Record<string, number>;
  alerts_last_7d: { threshold: number; hrv_watch: number; hrv_alert: number };
  top_alert_reasons_30d: { reason: string; count: number }[];
  recent_hrv_alerts: { id: number; user: string; owner: string; session: string; severity: string; score: number; emailed: boolean; created_at: string }[];
}

const DOC_TYPE_LABELS: Record<string, string> = {
  lab_report: 'Lab Reports', prescription: 'Prescriptions', medication: 'Medication Lists',
  imaging: 'Imaging', sensor_data: 'Sensor Data', other: 'Other',
};

function StatCard({ label, value, color }: { label: string; value: number | string; color: string }) {
  return (
    <div className="glass-panel" style={{ padding: '1.1rem' }}>
      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>{label}</p>
      <p style={{ fontWeight: 700, fontSize: '1.75rem', color }}>{value}</p>
    </div>
  );
}

export default function AdminOverviewPage() {
  const router = useRouter();
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState('');
  const [aiStatus, setAiStatus] = useState<'checking' | 'up' | 'down'>('checking');

  useEffect(() => {
    api.get('/admin/overview/').then((r) => setData(r.data)).catch((err) => {
      if (err.response?.status === 401) router.push('/login');
      else if (err.response?.status === 403) setError('Admin access required.');
      else setError('Failed to load overview.');
    });

    const AI_API_URL = process.env.NEXT_PUBLIC_AI_API_URL || 'http://127.0.0.1:8001';
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    fetch(`${AI_API_URL}/health`, { signal: controller.signal })
      .then((r) => setAiStatus(r.ok ? 'up' : 'down'))
      .catch(() => setAiStatus('down'))
      .finally(() => clearTimeout(timer));
  }, [router]);

  if (error) return <div className="container" style={{ padding: '4rem', textAlign: 'center', color: 'var(--error)' }}>{error}</div>;
  if (!data) return <div className="container" style={{ padding: '4rem', textAlign: 'center' }}>Loading platform overview…</div>;

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><LayoutDashboard size={24} color="var(--primary)" /> Platform Overview</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>Adoption, alert volume, and population health signals across all patients.</p>
        </div>
        <Link href="/admin/patients" className="btn btn-outline" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Users size={16} /> Patient List
        </Link>
      </div>

      {/* System status */}
      <div className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600, fontSize: '0.85rem' }}>
          <span style={{
            width: 9, height: 9, borderRadius: '50%',
            background: aiStatus === 'up' ? 'var(--success)' : aiStatus === 'down' ? 'var(--error)' : '#f59e0b',
            display: 'inline-block',
          }} />
          AI service (chat/trends/HRV): {aiStatus === 'checking' ? 'checking…' : aiStatus === 'up' ? 'online' : 'unreachable'}
        </span>
        {aiStatus === 'down' && (
          <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            HRV insights, forecasting, and chat will be unavailable to users until this is back up.
          </span>
        )}
      </div>

      {/* Top-level stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1rem' }}>
        <StatCard label="Total patients" value={data.users.total} color="var(--primary)" />
        <StatCard label="With sessions" value={data.users.with_sessions} color="#0d9488" />
        <StatCard label="Consent submitted" value={data.users.with_consent} color="var(--success)" />
        <StatCard label="Total sessions" value={data.sessions.total} color="var(--primary)" />
        <StatCard label="Devices online" value={`${data.devices.online} / ${data.devices.total}`} color={data.devices.online > 0 ? 'var(--success)' : 'var(--text-muted)'} />
      </div>

      {/* Alert volume */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
          <Bell size={20} color="var(--primary)" /> Alert Volume (last 7 days)
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1rem' }}>
          <StatCard label="Threshold alerts" value={data.alerts_last_7d.threshold} color="#f59e0b" />
          <StatCard label="HRV — worth watching" value={data.alerts_last_7d.hrv_watch} color="#f59e0b" />
          <StatCard label="HRV — alert (emailed)" value={data.alerts_last_7d.hrv_alert} color="var(--error)" />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>

        {/* Population health signal */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
            <TrendingUp size={20} color="var(--primary)" /> Population Health Signals (30d)
          </h3>
          {data.top_alert_reasons_30d.length === 0 ? (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No anomaly alerts in the last 30 days.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              {data.top_alert_reasons_30d.map((r) => (
                <div key={r.reason} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div style={{ flex: 1, height: 8, borderRadius: 4, background: 'var(--panel-bg-light)', overflow: 'hidden' }}>
                    <div style={{
                      width: `${Math.min(100, (r.count / data.top_alert_reasons_30d[0].count) * 100)}%`,
                      height: '100%', background: 'var(--primary)',
                    }} />
                  </div>
                  <span style={{ fontSize: '0.8rem', minWidth: 160 }}>{r.reason}</span>
                  <strong style={{ fontSize: '0.85rem' }}>{r.count}</strong>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Document categories */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
            <FileText size={20} color="var(--primary)" /> Uploaded Documents by Category
          </h3>
          {Object.keys(data.documents_by_type).length === 0 ? (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No documents uploaded yet.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {Object.entries(data.documents_by_type).map(([type, count]) => (
                <div key={type} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                  <span>{DOC_TYPE_LABELS[type] || type}</span>
                  <strong>{count}</strong>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Recent HRV alerts across all patients */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
          <HeartPulse size={20} color="var(--primary)" /> Recent HRV Alerts (all patients)
        </h3>
        {data.recent_hrv_alerts.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No HRV anomaly alerts yet.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {data.recent_hrv_alerts.map((a) => (
              <div key={a.id} style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.75rem 0', borderBottom: '1px solid var(--border)', flexWrap: 'wrap' }}>
                <span style={{
                  fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', padding: '0.15rem 0.5rem', borderRadius: '999px',
                  color: a.severity === 'alert' ? 'var(--error)' : '#f59e0b',
                  background: a.severity === 'alert' ? 'rgba(239,68,68,0.1)' : 'rgba(245,158,11,0.1)',
                }}>{a.severity}</span>
                <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>{a.user}</span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>score {a.score.toFixed(2)}</span>
                {a.emailed && <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}><Mail size={11} /> emailed</span>}
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{new Date(a.created_at).toLocaleString()}</span>
                <Link href={`/portal/${encodeURIComponent(a.owner)}/${encodeURIComponent(a.session)}`}
                  style={{ marginLeft: 'auto', fontSize: '0.8rem', color: 'var(--primary)', fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                  View <ChevronRight size={14} />
                </Link>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
