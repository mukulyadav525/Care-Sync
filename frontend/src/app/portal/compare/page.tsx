"use client";

import { useEffect, useState, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { GitCompare, ArrowLeft, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import api from '@/lib/api';

interface SignalStats {
  label: string; unit: string; color: string;
  min: number; max: number; avg: number; std: number;
}
interface SessionResult { owner: string; name: string; stats: Record<string, SignalStats> }
interface CompareResult { a: SessionResult; b: SessionResult }

const SIGNALS = ['HR', 'EDA', 'TEMP', 'ACC'];

function DiffBadge({ a, b }: { a: number; b: number }) {
  const diff = b - a;
  const pct = a !== 0 ? (diff / Math.abs(a)) * 100 : 0;
  if (Math.abs(pct) < 1) return <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}><Minus size={12} /> ~same</span>;
  const up = diff > 0;
  return (
    <span style={{ color: up ? '#ef4444' : '#10b981', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: 2 }}>
      {up ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
      {up ? '+' : ''}{pct.toFixed(1)}%
    </span>
  );
}

function CompareContent() {
  const router = useRouter();
  const params = useSearchParams();
  const [sessionA, setSessionA] = useState(params.get('a') || '');
  const [sessionB, setSessionB] = useState(params.get('b') || '');
  const [result, setResult] = useState<CompareResult | null>(null);
  const [sessions, setSessions] = useState<{ owner: string; name: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    api.get('/device/sessions/').then((r) => setSessions(r.data.sessions || []))
      .catch(() => router.push('/login'));
  }, [router]);

  // Auto-compare if both params supplied via URL
  useEffect(() => {
    if (params.get('a') && params.get('b')) handleCompare();
  }, []); // eslint-disable-line

  const handleCompare = async () => {
    if (!sessionA || !sessionB) { setError('Select both sessions.'); return; }
    setError('');
    setLoading(true);
    try {
      const res = await api.get(`/device/sessions/compare/?a=${encodeURIComponent(sessionA)}&b=${encodeURIComponent(sessionB)}`);
      setResult(res.data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Comparison failed.');
    } finally { setLoading(false); }
  };

  const sessionLabel = (s: string) => s || '— select —';
  const sessionOptions = sessions.map((s) => `${s.owner}/${s.name}`);

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem', maxWidth: 860 }}>
      <div>
        <Link href="/portal" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
          <ArrowLeft size={15} /> Sessions
        </Link>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><GitCompare size={24} color="var(--primary)" /> Compare Sessions</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>Compare signal averages between two sessions side-by-side.</p>
      </div>

      {/* Session picker */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '1rem', alignItems: 'end' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Session A</label>
            <select value={sessionA} onChange={(e) => setSessionA(e.target.value)}
              style={{ width: '100%', padding: '0.65rem 0.75rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontSize: '0.9rem' }}>
              <option value="">— select session —</option>
              {sessionOptions.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <span style={{ textAlign: 'center', fontWeight: 700, color: 'var(--text-muted)', paddingBottom: 8 }}>vs</span>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Session B</label>
            <select value={sessionB} onChange={(e) => setSessionB(e.target.value)}
              style={{ width: '100%', padding: '0.65rem 0.75rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontSize: '0.9rem' }}>
              <option value="">— select session —</option>
              {sessionOptions.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>
        {error && <p style={{ color: 'var(--error)', fontSize: '0.85rem', marginTop: '0.75rem' }}>{error}</p>}
        <button onClick={handleCompare} className="btn btn-primary" disabled={loading} style={{ marginTop: '1rem' }}>
          {loading ? 'Comparing…' : 'Compare'}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Column headers */}
          <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr 1fr 80px', gap: '1rem', padding: '0.5rem 1.25rem', fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            <span>Signal</span>
            <span style={{ textAlign: 'center' }}>{result.a.name}</span>
            <span style={{ textAlign: 'center' }}>{result.b.name}</span>
            <span style={{ textAlign: 'center' }}>Δ B vs A</span>
          </div>

          {SIGNALS.map((key) => {
            const sA = result.a.stats[key];
            const sB = result.b.stats[key];
            if (!sA && !sB) return null;
            return (
              <div key={key} className="glass-panel" style={{ padding: '1.1rem 1.25rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr 1fr 80px', gap: '1rem', alignItems: 'center' }}>
                  <div>
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: sA?.color || sB?.color, display: 'inline-block', marginRight: 8 }} />
                    <strong style={{ fontSize: '0.9rem' }}>{sA?.label || sB?.label}</strong>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{sA?.unit || sB?.unit}</p>
                  </div>
                  <StatCell s={sA} />
                  <StatCell s={sB} />
                  <div style={{ textAlign: 'center' }}>
                    {sA && sB ? <DiffBadge a={sA.avg} b={sB.avg} /> : <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>—</span>}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function StatCell({ s }: { s?: SignalStats }) {
  if (!s) return <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>No data</div>;
  return (
    <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', textAlign: 'center' }}>
      {[['Min', s.min], ['Avg', s.avg], ['Max', s.max]].map(([label, val]) => (
        <div key={label as string}>
          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{label}</p>
          <p style={{ fontWeight: 700, fontSize: '0.95rem' }}>{(val as number).toFixed(1)}</p>
        </div>
      ))}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="container" style={{ padding: '4rem', textAlign: 'center' }}>Loading…</div>}>
      <CompareContent />
    </Suspense>
  );
}
