"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Bell, Plus, Trash2, ToggleLeft, ToggleRight, History, AlertTriangle, Link as LinkIcon } from 'lucide-react';
import Link from 'next/link';
import api from '@/lib/api';

type Operator = 'gt' | 'lt';
type Signal = 'HR' | 'EDA' | 'TEMP' | 'ACC';

interface AlertRule {
  id: number;
  signal: Signal;
  operator: Operator;
  threshold: number;
  label: string;
  enabled: boolean;
  created_at: string;
}

const SIGNAL_OPTIONS: { value: Signal; label: string; unit: string }[] = [
  { value: 'HR', label: 'Heart Rate', unit: 'bpm' },
  { value: 'EDA', label: 'Skin Conductance', unit: 'µS' },
  { value: 'TEMP', label: 'Skin Temperature', unit: '°C' },
  { value: 'ACC', label: 'Movement Magnitude', unit: 'g' },
];

const SIGNAL_COLORS: Record<Signal, string> = {
  HR: '#ef4444', EDA: '#0d9488', TEMP: '#f59e0b', ACC: '#10b981',
};

interface FiredHistory {
  id: number; label: string; signal: string; operator: string; threshold: number;
  actual_mean: number; owner: string; session: string; fired_at: string;
}

export default function AlertsPage() {
  const router = useRouter();
  const [tab, setTab] = useState<'rules' | 'history'>('rules');
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [history, setHistory] = useState<FiredHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState<{ signal: Signal; operator: Operator; threshold: string; label: string }>({
    signal: 'HR', operator: 'gt', threshold: '', label: '',
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([api.get('/alerts/'), api.get('/alerts/history/')])
      .then(([r, h]) => { setRules(r.data.rules); setHistory(h.data.history); setLoading(false); })
      .catch(() => { router.push('/login'); });
  }, [router]);

  const clearHistory = async () => {
    if (!confirm('Clear all alert history?')) return;
    await api.delete('/alerts/history/clear/');
    setHistory([]);
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSaving(true);
    try {
      const res = await api.post('/alerts/', {
        signal: form.signal,
        operator: form.operator,
        threshold: parseFloat(form.threshold),
        label: form.label,
      });
      setRules((prev) => [...prev, res.data]);
      setShowForm(false);
      setForm({ signal: 'HR', operator: 'gt', threshold: '', label: '' });
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to create rule.');
    } finally { setSaving(false); }
  };

  const handleDelete = async (id: number) => {
    await api.delete(`/alerts/${id}/`);
    setRules((prev) => prev.filter((r) => r.id !== id));
  };

  const handleToggle = async (rule: AlertRule) => {
    const res = await api.patch(`/alerts/${rule.id}/`, { enabled: !rule.enabled });
    setRules((prev) => prev.map((r) => r.id === rule.id ? res.data : r));
  };

  const signalUnit = (s: Signal) => SIGNAL_OPTIONS.find((o) => o.value === s)?.unit || '';

  if (loading) return <div className="container" style={{ padding: '4rem', textAlign: 'center' }}>Loading…</div>;

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem', maxWidth: 720 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Bell size={24} color="var(--primary)" /> Threshold Alerts</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            Alerts fire when a session's average value exceeds your thresholds.
          </p>
        </div>
        {tab === 'rules' && (
          <button onClick={() => setShowForm((v) => !v)} className="btn btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Plus size={16} /> New Rule
          </button>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '2px solid var(--border)' }}>
        {([['rules', 'Alert Rules', <Bell size={15} />], ['history', `History${history.length ? ` (${history.length})` : ''}`, <History size={15} />]] as const).map(([key, label, icon]) => (
          <button key={key} onClick={() => setTab(key)}
            style={{ padding: '0.65rem 1.25rem', border: 'none', background: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '0.4rem', borderBottom: tab === key ? '2px solid var(--primary)' : '2px solid transparent', marginBottom: -2, color: tab === key ? 'var(--primary)' : 'var(--text-muted)' }}>
            {icon}{label}
          </button>
        ))}
      </div>

      {/* History tab */}
      {tab === 'history' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {history.length > 0 && (
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button onClick={clearHistory} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Trash2 size={13} /> Clear history
              </button>
            </div>
          )}
          {history.length === 0 ? (
            <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              <History size={36} style={{ marginBottom: '1rem', opacity: 0.25 }} />
              <p>No alerts have fired yet. Open a session that breaches a rule to see history here.</p>
            </div>
          ) : (
            history.map((h) => (
              <div key={h.id} className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
                <AlertTriangle size={16} color="#f59e0b" style={{ flexShrink: 0 }} />
                <div style={{ flex: 1 }}>
                  <p style={{ fontWeight: 600, fontSize: '0.9rem' }}>{h.label}</p>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    avg {h.signal} = {h.actual_mean} &nbsp;·&nbsp; {new Date(h.fired_at).toLocaleString()}
                  </p>
                </div>
                <Link href={`/portal/${encodeURIComponent(h.owner)}/${encodeURIComponent(h.session)}`}
                  style={{ fontSize: '0.8rem', color: 'var(--primary)', fontWeight: 600, whiteSpace: 'nowrap' }}>
                  {h.session} →
                </Link>
              </div>
            ))
          )}
        </div>
      )}

      {tab === 'rules' && showForm && (
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>New alert rule</h3>
          {error && <p style={{ color: 'var(--error)', fontSize: '0.85rem', marginBottom: '1rem' }}>{error}</p>}
          <form onSubmit={handleCreate} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Signal</label>
                <select value={form.signal} onChange={(e) => setForm({ ...form, signal: e.target.value as Signal })}
                  style={{ width: '100%', padding: '0.6rem 0.75rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontSize: '0.9rem' }}>
                  {SIGNAL_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Condition</label>
                <select value={form.operator} onChange={(e) => setForm({ ...form, operator: e.target.value as Operator })}
                  style={{ width: '100%', padding: '0.6rem 0.75rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontSize: '0.9rem' }}>
                  <option value="gt">Greater than (&gt;)</option>
                  <option value="lt">Less than (&lt;)</option>
                </select>
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Threshold ({signalUnit(form.signal)})</label>
                <input type="number" step="0.1" required value={form.threshold} onChange={(e) => setForm({ ...form, threshold: e.target.value })}
                  placeholder="e.g. 110"
                  style={{ width: '100%', padding: '0.6rem 0.75rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontSize: '0.9rem' }} />
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Label (optional)</label>
                <input type="text" value={form.label} onChange={(e) => setForm({ ...form, label: e.target.value })}
                  placeholder="e.g. High HR alert"
                  style={{ width: '100%', padding: '0.6rem 0.75rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontSize: '0.9rem' }} />
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button type="submit" className="btn btn-primary" disabled={saving}>{saving ? 'Saving…' : 'Create Rule'}</button>
              <button type="button" onClick={() => setShowForm(false)} className="btn btn-outline">Cancel</button>
            </div>
          </form>
        </div>
      )}

      {rules.length === 0 && !showForm && (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <Bell size={40} style={{ marginBottom: '1rem', opacity: 0.3 }} />
          <p>No alert rules yet. Create one to get notified when a session breaches a threshold.</p>
        </div>
      )}

      {rules.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {rules.map((rule) => {
            const sig = SIGNAL_OPTIONS.find((o) => o.value === rule.signal);
            const color = SIGNAL_COLORS[rule.signal];
            return (
              <div key={rule.id} className="glass-panel" style={{ padding: '1.1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem', opacity: rule.enabled ? 1 : 0.55 }}>
                <div style={{ width: 4, height: 40, borderRadius: 2, background: color, flexShrink: 0 }} />
                <div style={{ flex: 1 }}>
                  <p style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                    {rule.label || `${sig?.label} ${rule.operator === 'gt' ? '>' : '<'} ${rule.threshold} ${sig?.unit}`}
                  </p>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                    {sig?.label} · avg {rule.operator === 'gt' ? 'above' : 'below'} {rule.threshold} {sig?.unit}
                  </p>
                </div>
                <span style={{ fontSize: '0.75rem', color: rule.enabled ? 'var(--success)' : 'var(--text-muted)', fontWeight: 600 }}>
                  {rule.enabled ? 'Active' : 'Paused'}
                </span>
                <button onClick={() => handleToggle(rule)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: rule.enabled ? 'var(--primary)' : 'var(--text-muted)', display: 'flex' }}>
                  {rule.enabled ? <ToggleRight size={22} /> : <ToggleLeft size={22} />}
                </button>
                <button onClick={() => handleDelete(rule.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', display: 'flex' }}>
                  <Trash2 size={16} />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
