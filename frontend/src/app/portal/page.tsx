"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Activity, Clock, Radio, Tag, ChevronRight, HeartPulse } from 'lucide-react';
import api from '@/lib/api';

interface Session {
  owner: string;
  name: string;
  signals: string[];
  has_tags: boolean;
  start: string | null;
  end: string | null;
  duration_sec: number;
}

function formatDuration(sec: number): string {
  if (!sec) return '—';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

function formatDate(iso: string | null): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: 'medium', timeStyle: 'short',
  });
}

export default function Portal() {
  const router = useRouter();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const res = await api.get('/device/sessions/');
        setSessions(res.data.sessions || []);
      } catch (err: any) {
        if (err.response?.status === 401) { router.push('/login'); return; }
        setError(err.response?.data?.error || 'Failed to load sessions.');
      } finally {
        setLoading(false);
      }
    };
    fetchSessions();
  }, [router]);

  if (loading) {
    return <div className="container" style={{ padding: '4rem', textAlign: 'center' }}>Loading device sessions…</div>;
  }

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Radio size={26} color="var(--primary)" /> Signal Portal
        </h2>
        <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          Recorded sessions from your wearable device. Open one to explore heart rate,
          skin conductance, temperature, movement, BVP and HRV — by hour, day or week.
        </p>
      </div>

      {error && (
        <div style={{ padding: '0.75rem', backgroundColor: 'rgba(239,68,68,0.1)', color: 'var(--error)', borderRadius: 'var(--radius-sm)' }}>
          {error}
        </div>
      )}

      {sessions.length === 0 && !error ? (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <HeartPulse size={40} color="var(--border)" style={{ marginBottom: '1rem' }} />
          <h3 style={{ marginBottom: '0.5rem' }}>No sessions yet</h3>
          <p>Upload a device session folder (ACC, BVP, EDA, HR, IBI, TEMP …) into your user directory to see it here.</p>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.5rem' }}>
          {sessions.map((s) => (
            <Link
              key={`${s.owner}/${s.name}`}
              href={`/portal/${encodeURIComponent(s.owner)}/${encodeURIComponent(s.name)}`}
              className="glass-panel"
              style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem', color: 'inherit' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h3 style={{ fontSize: '1.1rem' }}>{s.name}</h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.25rem' }}>
                    {s.owner}
                  </p>
                </div>
                <ChevronRight size={20} color="var(--text-muted)" />
              </div>

              <div style={{ display: 'flex', gap: '1.25rem', flexWrap: 'wrap', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <Clock size={15} /> {formatDuration(s.duration_sec)}
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <Activity size={15} /> {s.signals.length} signals
                </span>
                {s.has_tags && (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <Tag size={15} /> tagged
                  </span>
                )}
              </div>

              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                {formatDate(s.start)}
              </div>

              <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                {s.signals.map((sig) => (
                  <span key={sig} style={{
                    fontSize: '0.7rem', fontWeight: 600, padding: '0.2rem 0.55rem',
                    borderRadius: '999px', backgroundColor: 'var(--panel-bg-light)',
                    color: 'var(--primary-hover)', border: '1px solid var(--panel-border)',
                  }}>
                    {sig}
                  </span>
                ))}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
