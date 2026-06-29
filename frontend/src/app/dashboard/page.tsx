"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Activity, FileText, Upload, ShieldCheck, Radio, Wifi, WifiOff, ChevronRight, Clock, Bluetooth, HeartPulse, Thermometer, Droplets } from 'lucide-react';
import api from '@/lib/api';

const BLE_STORAGE_KEY = 'ble_latest';
const BLE_STALE_MS = 10_000; // hide tile if no update in 10s

interface BleLatest { t: number; HR?: number; EDA?: number; TEMP?: number; ACC?: number; }

function fmtDuration(sec: number): string {
  if (!sec) return '—';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

export default function Dashboard() {
  const router = useRouter();
  const [user, setUser] = useState<any>(null);
  const [profile, setProfile] = useState<any>(null);
  const [devices, setDevices] = useState<{ online: number; total: number; list: any[] }>({ online: 0, total: 0, list: [] });
  const [sessions, setSessions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [bleData, setBleData] = useState<BleLatest | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [p, d, s] = await Promise.allSettled([
          api.get('/profile/'),
          api.get('/devices/'),
          api.get('/device/sessions/'),
        ]);
        if (p.status === 'fulfilled') setProfile(p.value.data);
        else { router.push('/login'); return; }
        setUser(JSON.parse(localStorage.getItem('user') || '{}'));
        if (d.status === 'fulfilled') setDevices({ online: d.value.data.online, total: d.value.data.total, list: d.value.data.devices });
        if (s.status === 'fulfilled') setSessions(s.value.data.sessions || []);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [router]);

  // Poll localStorage for live BLE data written by the Devices page
  useEffect(() => {
    const check = () => {
      try {
        const raw = localStorage.getItem(BLE_STORAGE_KEY);
        if (!raw) { setBleData(null); return; }
        const d: BleLatest = JSON.parse(raw);
        if (Date.now() - d.t > BLE_STALE_MS) { setBleData(null); return; }
        setBleData(d);
      } catch { setBleData(null); }
    };
    check();
    const t = setInterval(check, 2000);
    return () => clearInterval(t);
  }, []);

  if (loading) return <div className="container" style={{ padding: '4rem', textAlign: 'center' }}>Loading dashboard...</div>;

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <h2>Welcome, {user?.username}</h2>
        <Link href="/files" className="btn btn-primary">
          <FileText size={18} /> View My Files
        </Link>
      </div>

      {/* Live BLE strip — only shown when device is streaming */}
      {bleData && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', padding: '0.85rem 1.25rem', background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.25)', borderRadius: 'var(--radius-md)', flexWrap: 'wrap' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 700, fontSize: '0.85rem', color: 'var(--primary)' }}>
            <Bluetooth size={15} /> Live BLE
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981', display: 'inline-block', animation: 'pulse 1.5s infinite' }} />
          </span>
          {[
            { key: 'HR', label: 'HR', unit: 'bpm', icon: HeartPulse, color: '#ef4444' },
            { key: 'EDA', label: 'EDA', unit: 'µS', icon: Droplets, color: '#0d9488' },
            { key: 'TEMP', label: 'Temp', unit: '°C', icon: Thermometer, color: '#f59e0b' },
          ].map(({ key, label, unit, icon: Icon, color }) => {
            const val = (bleData as any)[key];
            if (val === undefined) return null;
            return (
              <span key={key} style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.9rem' }}>
                <Icon size={14} color={color} />
                <strong style={{ color }}>{val.toFixed(1)}</strong>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>{unit}</span>
              </span>
            );
          })}
          <Link href="/devices" style={{ marginLeft: 'auto', fontSize: '0.8rem', color: 'var(--primary)' }}>Open BLE →</Link>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>

        {/* Profile status */}
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
            <Activity size={24} color="var(--primary)" /> Profile Status
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Email</span><strong>{user?.email}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Role</span><strong>{user?.is_superuser ? 'Admin' : 'Patient'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Consent Form</span>
              {profile?.form_submitted
                ? <strong style={{ color: 'var(--success)' }}>Submitted</strong>
                : <strong style={{ color: 'var(--error)' }}>Pending</strong>}
            </div>
          </div>
        </div>

        {/* Device status */}
        <Link href="/devices" className="glass-panel" style={{ padding: '2rem', color: 'inherit', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Radio size={24} color="var(--primary)" /> Devices</span>
            <ChevronRight size={18} color="var(--text-muted)" />
          </h3>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginBottom: '1rem' }}>
            <span style={{ fontSize: '2rem', fontWeight: 700, color: devices.online ? 'var(--success)' : 'var(--text-muted)' }}>{devices.online}</span>
            <span style={{ color: 'var(--text-muted)' }}>of {devices.total} online</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {devices.list.slice(0, 3).map((d: any) => (
              <div key={d.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  {d.is_online ? <Wifi size={14} color="var(--success)" /> : <WifiOff size={14} color="var(--text-muted)" />}
                  {d.name}
                </span>
                <span style={{ color: 'var(--text-muted)' }}>{d.battery != null ? `${d.battery}%` : ''}</span>
              </div>
            ))}
            {devices.total === 0 && <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No devices registered yet.</p>}
          </div>
        </Link>

        {/* Quick actions */}
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h3 style={{ marginBottom: '1.5rem' }}>Quick Actions</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <Link href="/files" className="btn btn-outline" style={{ justifyContent: 'flex-start' }}>
              <Upload size={18} /> Upload New Data
            </Link>
            {!profile?.form_submitted && (
              <Link href="/consent" className="btn btn-outline" style={{ justifyContent: 'flex-start', color: 'var(--error)', borderColor: 'var(--error)' }}>
                <ShieldCheck size={18} /> Submit Consent Form
              </Link>
            )}
            <Link href="/portal" className="btn btn-outline" style={{ justifyContent: 'flex-start' }}>
              <Activity size={18} /> Open Signal Portal
            </Link>
          </div>
        </div>
      </div>

      {/* Recent sessions */}
      <div className="glass-panel" style={{ padding: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Radio size={22} color="var(--primary)" /> Recent Sessions</h3>
          <Link href="/portal" style={{ fontSize: '0.85rem', fontWeight: 600 }}>View all →</Link>
        </div>
        {sessions.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>No recorded sessions yet.</p>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '1rem' }}>
            {sessions.slice(0, 4).map((s: any) => (
              <Link key={`${s.owner}/${s.name}`} href={`/portal/${encodeURIComponent(s.owner)}/${encodeURIComponent(s.name)}`}
                style={{ padding: '1rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', color: 'inherit', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <strong style={{ fontSize: '0.95rem' }}>{s.name}</strong>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  <Clock size={13} /> {fmtDuration(s.duration_sec)} · {s.signals.length} signals
                </span>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
