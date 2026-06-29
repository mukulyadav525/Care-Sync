"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Lock, ShieldCheck, Radio, User as UserIcon, ChevronRight } from 'lucide-react';
import api from '@/lib/api';

export default function Settings() {
  const router = useRouter();
  const [user, setUser] = useState<any>(null);

  const [pwd, setPwd] = useState({ old_password: '', new_password: '', confirm: '' });
  const [msg, setMsg] = useState('');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const u = localStorage.getItem('user');
    if (!u) { router.push('/login'); return; }
    setUser(JSON.parse(u));
  }, [router]);

  const changePassword = async (e: React.FormEvent) => {
    e.preventDefault(); setMsg(''); setError('');
    if (pwd.new_password !== pwd.confirm) { setError('New passwords do not match.'); return; }
    setSaving(true);
    try {
      await api.post('/auth/change-password/', { old_password: pwd.old_password, new_password: pwd.new_password });
      setMsg('Password updated successfully.');
      setPwd({ old_password: '', new_password: '', confirm: '' });
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to update password.');
    } finally { setSaving(false); }
  };

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', maxWidth: 760 }}>
      <div>
        <h2>Settings</h2>
        <p style={{ color: 'var(--text-muted)', marginTop: '0.4rem' }}>Manage your account and security.</p>
      </div>

      {/* Account */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', fontSize: '1.05rem' }}><UserIcon size={18} color="var(--primary)" /> Account</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.9rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Username</span><strong>{user?.username}</strong></div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Email</span><strong>{user?.email}</strong></div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Role</span><strong>{user?.is_superuser ? 'Admin' : 'Patient'}</strong></div>
        </div>
      </div>

      {/* 2FA status */}
      <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <div style={{ background: 'rgba(16,185,129,0.12)', padding: '0.85rem', borderRadius: '50%', display: 'flex' }}>
          <ShieldCheck size={22} color="var(--primary)" />
        </div>
        <div>
          <h3 style={{ fontSize: '1.05rem' }}>Two-factor authentication</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Enabled — every sign-in requires an email code. This is mandatory for medical data.</p>
        </div>
      </div>

      {/* Change password */}
      <form onSubmit={changePassword} className="glass-panel" style={{ padding: '1.5rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem', fontSize: '1.05rem' }}><Lock size={18} color="var(--primary)" /> Change password</h3>

        {msg && <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(22,163,74,0.1)', color: 'var(--success)', borderRadius: 'var(--radius-sm)', marginBottom: '1rem', fontSize: '0.85rem' }}>{msg}</div>}
        {error && <div style={{ padding: '0.6rem 0.75rem', background: 'rgba(239,68,68,0.1)', color: 'var(--error)', borderRadius: 'var(--radius-sm)', marginBottom: '1rem', fontSize: '0.85rem' }}>{error}</div>}

        <div className="input-group">
          <label className="input-label">Current password</label>
          <input type="password" className="input-field" value={pwd.old_password} onChange={e => setPwd({ ...pwd, old_password: e.target.value })} required />
        </div>
        <div className="input-group">
          <label className="input-label">New password</label>
          <input type="password" className="input-field" placeholder="At least 8 chars, not all numbers" value={pwd.new_password} onChange={e => setPwd({ ...pwd, new_password: e.target.value })} required />
        </div>
        <div className="input-group">
          <label className="input-label">Confirm new password</label>
          <input type="password" className="input-field" value={pwd.confirm} onChange={e => setPwd({ ...pwd, confirm: e.target.value })} required />
        </div>
        <button type="submit" className="btn btn-primary" disabled={saving}>{saving ? 'Updating…' : 'Update password'}</button>
      </form>

      {/* Devices link */}
      <Link href="/devices" className="glass-panel" style={{ padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'inherit' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Radio size={18} color="var(--primary)" /> Manage connected devices</span>
        <ChevronRight size={18} color="var(--text-muted)" />
      </Link>
    </div>
  );
}
