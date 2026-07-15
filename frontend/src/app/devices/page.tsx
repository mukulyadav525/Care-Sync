"use client";

import React, { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Radio, Battery, Cpu, Plus, Trash2, Wifi, WifiOff, KeyRound, Copy, Check,
  Bluetooth, BluetoothConnected, BluetoothOff, Circle, Settings, Download,
} from 'lucide-react';
import api from '@/lib/api';

/* ─────────────────────────────────────────────
   Registered device types
───────────────────────────────────────────── */
interface Device {
  id: number;
  device_id: string;
  name: string;
  firmware: string;
  battery: number | null;
  current_session: string;
  last_seen: string | null;
  is_online: boolean;
}

function lastSeen(iso: string | null): string {
  if (!iso) return 'never';
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

/* ─────────────────────────────────────────────
   BLE types
───────────────────────────────────────────── */
const DEFAULT_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
const DEFAULT_NOTIFY_UUID  = '6e400003-b5a3-f393-e0a9-e50e24dcca9e';
const MAX_LIVE_POINTS = 300;
const BLE_STORAGE_KEY = 'ble_latest';

interface BleSample {
  t: number;
  HR?: number; EDA?: number; TEMP?: number; ACC?: number; raw?: string;
}

function parseSample(raw: DataView): Partial<BleSample> {
  try {
    const text = new TextDecoder().decode(raw.buffer);
    const obj = JSON.parse(text);
    const acc = obj.ACC;
    return {
      HR: typeof obj.HR === 'number' ? obj.HR : undefined,
      EDA: typeof obj.EDA === 'number' ? obj.EDA : undefined,
      TEMP: typeof obj.TEMP === 'number' ? obj.TEMP : undefined,
      ACC: Array.isArray(acc)
        ? Math.sqrt(acc.reduce((s: number, v: number) => s + v * v, 0))
        : typeof acc === 'number' ? acc : undefined,
    };
  } catch {
    return { raw: new TextDecoder().decode(raw.buffer) };
  }
}

const BLE_SIGNALS: { key: keyof Omit<BleSample, 't' | 'raw'>; label: string; unit: string; color: string }[] = [
  { key: 'HR',   label: 'Heart Rate',       unit: 'bpm', color: '#ef4444' },
  { key: 'EDA',  label: 'Skin Conductance', unit: 'µS',  color: '#0d9488' },
  { key: 'TEMP', label: 'Temperature',      unit: '°C',  color: '#f59e0b' },
  { key: 'ACC',  label: 'Movement',         unit: 'g',   color: '#10b981' },
];

/* ─────────────────────────────────────────────
   Main page
───────────────────────────────────────────── */
export default function DevicesPage() {
  const router = useRouter();
  const [tab, setTab] = useState<'registered' | 'ble'>('registered');

  /* ── Registered devices state ── */
  const [devices, setDevices] = useState<Device[]>([]);
  const [devLoading, setDevLoading] = useState(true);
  const [devError, setDevError] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ device_id: '', name: '' });
  const [newKey, setNewKey] = useState('');
  const [copied, setCopied] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  /* ── BLE state ── */
  const [bleStatus, setBleStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');
  const [bleName, setBleName] = useState('');
  const [bleSamples, setBleSamples] = useState<BleSample[]>([]);
  const [bleLatest, setBleLatest] = useState<Partial<BleSample>>({});
  const [bleLog, setBleLog] = useState<string[]>([]);
  const [showBleSettings, setShowBleSettings] = useState(false);
  const [serviceUUID, setServiceUUID] = useState(DEFAULT_SERVICE_UUID);
  const [notifyUUID, setNotifyUUID] = useState(DEFAULT_NOTIFY_UUID);
  const [bleSessionName, setBleSessionName] = useState('');
  const [bleSaving, setBleSaving] = useState(false);
  const deviceRef = useRef<any>(null);
  const samplesRef = useRef<BleSample[]>([]);

  /* ── Simulated device (demo mode — no real hardware needed) ── */
  const [simulating, setSimulating] = useState(false);
  const simIntervalRef = useRef<any>(null);
  const simTickRef = useRef(0);

  const isBleSupported = typeof navigator !== 'undefined' && 'bluetooth' in navigator;

  /* ── Load registered devices ── */
  const fetchDevices = async () => {
    try {
      const res = await api.get('/devices/');
      setDevices(res.data.devices || []);
    } catch (err: any) {
      if (err.response?.status === 401) { router.push('/login'); return; }
      setDevError('Failed to load devices.');
    } finally { setDevLoading(false); }
  };

  useEffect(() => {
    fetchDevices();
    const t = setInterval(fetchDevices, 30000);
    return () => clearInterval(t);
  }, []);

  /* ── Register device ── */
  const register = async (e: React.FormEvent) => {
    e.preventDefault(); setDevError(''); setSubmitting(true);
    try {
      const res = await api.post('/devices/register/', form);
      setNewKey(res.data.key);
      setForm({ device_id: '', name: '' });
      setShowForm(false);
      fetchDevices();
    } catch (err: any) {
      setDevError(err.response?.data?.error || 'Failed to register device.');
    } finally { setSubmitting(false); }
  };

  const remove = async (id: number) => {
    if (!confirm('Remove this device? Its key will stop working.')) return;
    try { await api.delete(`/devices/${id}/delete/`); fetchDevices(); }
    catch { alert('Failed to remove device.'); }
  };

  const copyKey = () => {
    navigator.clipboard.writeText(newKey);
    setCopied(true); setTimeout(() => setCopied(false), 1500);
  };

  /* ── BLE helpers ── */
  const addBleLog = (msg: string) => setBleLog((p) => [`${new Date().toLocaleTimeString()} — ${msg}`, ...p.slice(0, 49)]);

  const handleBleConnect = async () => {
    if (!isBleSupported) { addBleLog('Web Bluetooth not supported. Use Chrome or Edge.'); return; }
    setBleStatus('connecting');
    addBleLog('Requesting BLE device…');
    try {
      const device = await (navigator as any).bluetooth.requestDevice({
        filters: [{ services: [serviceUUID] }],
        optionalServices: [serviceUUID],
      });
      deviceRef.current = device;
      setBleName(device.name || 'Unknown');
      addBleLog(`Paired: ${device.name}`);
      device.addEventListener('gattserverdisconnected', () => {
        setBleStatus('idle'); addBleLog('Device disconnected.');
        localStorage.removeItem(BLE_STORAGE_KEY);
      });
      const server = await device.gatt.connect();
      const service = await server.getPrimaryService(serviceUUID);
      const char = await service.getCharacteristic(notifyUUID);
      char.addEventListener('characteristicvaluechanged', (e: any) => {
        const parsed = parseSample(e.target.value as DataView);
        const sample: BleSample = { t: Date.now(), ...parsed };
        samplesRef.current = [...samplesRef.current.slice(-(MAX_LIVE_POINTS - 1)), sample];
        setBleSamples([...samplesRef.current]);
        setBleLatest(parsed);
        localStorage.setItem(BLE_STORAGE_KEY, JSON.stringify({ ...parsed, t: Date.now() }));
      });
      await char.startNotifications();
      addBleLog('Streaming live data…');
      setBleStatus('connected');
    } catch (err: any) {
      addBleLog(`Error: ${err.message || err}`);
      setBleStatus('error');
    }
  };

  const handleBleDisconnect = () => {
    if (deviceRef.current?.gatt?.connected) deviceRef.current.gatt.disconnect();
    setBleStatus('idle'); addBleLog('Disconnected.');
    localStorage.removeItem(BLE_STORAGE_KEY);
  };

  /* ── Simulate a device with synthetic vitals (no hardware required) ──
     Useful while real hardware is still in development: generates a
     realistic-looking HR/EDA/TEMP/ACC stream, including a deliberate
     "spike" partway through each loop so the anomaly/alert pipeline has
     something real to detect and demo end-to-end. Writes to the exact
     same BLE_STORAGE_KEY the real Bluetooth path uses, so the dashboard's
     live strip and this page's tiles behave identically either way. */
  const startSimulation = () => {
    if (simIntervalRef.current) return;
    setSimulating(true);
    setBleStatus('connected');
    setBleName('Simulated Watch (Demo Mode)');
    addBleLog('Started demo simulation — synthetic data, no real device connected.');
    simTickRef.current = 0;

    simIntervalRef.current = setInterval(() => {
      const t = simTickRef.current++;
      const loopPos = t % 90;
      const spike = loopPos > 40 && loopPos < 58; // ~18s simulated "event" each 90s loop

      const hr = 68 + 4 * Math.sin(t / 12) + (Math.random() - 0.5) * 3 + (spike ? 34 : 0);
      const eda = 1.8 + (Math.random() - 0.5) * 0.3 + (spike ? 1.3 : 0);
      const temp = 36.5 + (Math.random() - 0.5) * 0.1 + (spike ? 0.7 : 0);
      const acc = 0.05 + Math.random() * 0.05 + (spike ? 0.35 : 0);

      const parsed = {
        HR: Math.round(hr * 10) / 10,
        EDA: Math.round(eda * 100) / 100,
        TEMP: Math.round(temp * 10) / 10,
        ACC: Math.round(acc * 100) / 100,
      };
      const sample: BleSample = { t: Date.now(), ...parsed };
      samplesRef.current = [...samplesRef.current.slice(-(MAX_LIVE_POINTS - 1)), sample];
      setBleSamples([...samplesRef.current]);
      setBleLatest(parsed);
      localStorage.setItem(BLE_STORAGE_KEY, JSON.stringify({ ...parsed, t: Date.now() }));
    }, 1000);
  };

  const stopSimulation = () => {
    if (simIntervalRef.current) { clearInterval(simIntervalRef.current); simIntervalRef.current = null; }
    setSimulating(false);
    setBleStatus('idle');
    addBleLog('Stopped simulation.');
    localStorage.removeItem(BLE_STORAGE_KEY);
  };

  useEffect(() => () => { if (simIntervalRef.current) clearInterval(simIntervalRef.current); }, []);

  const handleBleExport = () => {
    if (!bleSamples.length) return;
    const keys = ['t', 'HR', 'EDA', 'TEMP', 'ACC'];
    const rows = bleSamples.map((s) => keys.map((k) => (s as any)[k] ?? '').join(','));
    const blob = new Blob([[keys.join(','), ...rows].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `ble_${Date.now()}.csv`;
    a.click(); URL.revokeObjectURL(url);
  };

  const handleBleSave = async () => {
    if (!bleSessionName.trim() || !bleSamples.length) return;
    setBleSaving(true);
    addBleLog(`Saving ${bleSamples.length} samples as "${bleSessionName}"…`);
    try {
      await api.post('/device/sessions/save-ble/', { session: bleSessionName.trim(), samples: samplesRef.current });
      addBleLog('Saved to portal.');
      setBleSessionName('');
    } catch (err: any) {
      addBleLog(`Save failed: ${err.response?.data?.error || err.message}`);
    } finally { setBleSaving(false); }
  };

  const bleStatusColor = { idle: 'var(--text-muted)', connecting: '#f59e0b', connected: '#10b981', error: '#ef4444' }[bleStatus];

  /* ─────────────── render ─────────────── */
  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>

      {/* Page header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Radio size={26} color="var(--primary)" /> Devices
          </h2>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.4rem' }}>
            Manage registered wearables and connect via Bluetooth.
          </p>
        </div>
        {tab === 'registered' && (
          <button onClick={() => { setShowForm(!showForm); setNewKey(''); }} className="btn btn-primary">
            <Plus size={18} /> Register device
          </button>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '2px solid var(--border)' }}>
        {(['registered', 'ble'] as const).map((key) => (
          <button key={key} onClick={() => setTab(key)}
            style={{ padding: '0.75rem 1.5rem', border: 'none', background: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem', borderBottom: tab === key ? '2px solid var(--primary)' : '2px solid transparent', marginBottom: -2, color: tab === key ? 'var(--primary)' : 'var(--text-muted)', transition: 'color 0.15s' }}>
            {key === 'registered' ? <Radio size={16} /> : <Bluetooth size={16} />}
            {key === 'registered' ? 'Connected Devices' : 'Bluetooth Live'}
          </button>
        ))}
      </div>

      {/* ═══ TAB: Registered devices ═══ */}
      {tab === 'registered' && (
        <>
          {devError && <div style={{ padding: '0.75rem', background: 'rgba(239,68,68,0.1)', color: 'var(--error)', borderRadius: 'var(--radius-sm)' }}>{devError}</div>}

          {newKey && (
            <div className="glass-panel" style={{ padding: '1.5rem', background: 'var(--panel-bg-light)', border: '1px solid var(--panel-border)' }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1rem', marginBottom: '0.5rem' }}>
                <KeyRound size={18} color="var(--primary-hover)" /> Device key — copy it now
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1rem' }}>
                Shown only once. Send as <code>X-Device-Key</code> header on heartbeats.
              </p>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <code style={{ flex: 1, padding: '0.75rem', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem', wordBreak: 'break-all' }}>{newKey}</code>
                <button onClick={copyKey} className="btn btn-outline" style={{ padding: '0.65rem' }}>
                  {copied ? <Check size={18} color="var(--success)" /> : <Copy size={18} />}
                </button>
              </div>
            </div>
          )}

          {showForm && (
            <form onSubmit={register} className="glass-panel" style={{ padding: '1.5rem', display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
              <div className="input-group" style={{ flex: 1, minWidth: 200, marginBottom: 0 }}>
                <label className="input-label">Device ID</label>
                <input className="input-field" placeholder="e.g. E4-001" value={form.device_id} onChange={e => setForm({ ...form, device_id: e.target.value })} required />
              </div>
              <div className="input-group" style={{ flex: 1, minWidth: 200, marginBottom: 0 }}>
                <label className="input-label">Display name</label>
                <input className="input-field" placeholder="e.g. Wrist E4" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
              </div>
              <button type="submit" className="btn btn-primary" disabled={submitting}>{submitting ? 'Saving…' : 'Save'}</button>
            </form>
          )}

          {devLoading ? (
            <p style={{ textAlign: 'center', color: 'var(--text-muted)' }}>Loading…</p>
          ) : devices.length === 0 ? (
            <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              <Radio size={40} color="var(--border)" style={{ marginBottom: '1rem' }} />
              <h3 style={{ marginBottom: '0.5rem' }}>No devices yet</h3>
              <p>Register your wearable to start tracking its status.</p>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.5rem' }}>
              {devices.map((d) => (
                <div key={d.id} className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <h3 style={{ fontSize: '1.1rem' }}>{d.name}</h3>
                      <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.2rem' }}>{d.device_id}</p>
                    </div>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem', fontWeight: 700, padding: '0.25rem 0.6rem', borderRadius: '999px', background: d.is_online ? 'rgba(22,163,74,0.12)' : 'var(--btn-toggle-bg)', color: d.is_online ? 'var(--success)' : 'var(--text-muted)' }}>
                      {d.is_online ? <Wifi size={13} /> : <WifiOff size={13} />}
                      {d.is_online ? 'Online' : 'Offline'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.85rem' }}>
                    <DevRow icon={<Battery size={15} />} label="Battery" value={d.battery != null ? `${d.battery}%` : '—'} />
                    <DevRow icon={<Cpu size={15} />} label="Firmware" value={d.firmware || '—'} />
                    <DevRow icon={<Radio size={15} />} label="Last seen" value={lastSeen(d.last_seen)} />
                    {d.current_session && <DevRow icon={<Radio size={15} />} label="Session" value={d.current_session} />}
                  </div>
                  <button onClick={() => remove(d.id)} className="btn btn-outline" style={{ alignSelf: 'flex-start', padding: '0.45rem 0.9rem', borderColor: 'var(--error)', color: 'var(--error)' }}>
                    <Trash2 size={15} /> Remove
                  </button>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* ═══ TAB: BLE Live Stream ═══ */}
      {tab === 'ble' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          {!isBleSupported && (
            <div style={{ padding: '1rem', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', borderRadius: 'var(--radius-md)', color: '#ef4444', fontSize: '0.9rem' }}>
              Web Bluetooth is not supported here. Use <strong>Chrome</strong> or <strong>Edge</strong> on desktop/Android.
            </div>
          )}

          {/* GATT settings */}
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button onClick={() => setShowBleSettings((v) => !v)} style={{ background: 'none', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '0.45rem 0.85rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
              <Settings size={14} /> GATT Config
            </button>
          </div>

          {showBleSettings && (
            <div className="glass-panel" style={{ padding: '1.25rem' }}>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>Match these UUIDs to your device firmware. Default: Nordic UART Service.</p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                {([['Service UUID', serviceUUID, setServiceUUID], ['Notify Characteristic UUID', notifyUUID, setNotifyUUID]] as [string, string, React.Dispatch<React.SetStateAction<string>>][]).map(([label, val, setter]) => (
                  <div key={label}>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>{label}</label>
                    <input value={val} onChange={(e) => setter(e.target.value)}
                      style={{ width: '100%', padding: '0.5rem 0.65rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontFamily: 'monospace', fontSize: '0.78rem' }} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Connection bar */}
          <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <div style={{ background: `${bleStatusColor}1a`, padding: '0.8rem', borderRadius: '50%', display: 'flex' }}>
                {bleStatus === 'connected' ? <BluetoothConnected size={22} color={bleStatusColor} /> : bleStatus === 'error' ? <BluetoothOff size={22} color={bleStatusColor} /> : <Bluetooth size={22} color={bleStatusColor} />}
              </div>
              <div>
                <p style={{ fontWeight: 600 }}>
                  {bleStatus === 'idle' ? 'Not connected' : bleStatus === 'connecting' ? 'Connecting…' : bleStatus === 'error' ? 'Connection failed' : `Connected — ${bleName}`}
                </p>
                {bleStatus === 'connected' && <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{bleSamples.length} samples{simulating ? ' · simulated' : ''}</p>}
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.65rem', flexWrap: 'wrap' }}>
              {bleStatus !== 'connected' ? (
                <>
                  <button onClick={handleBleConnect} className="btn btn-primary" disabled={bleStatus === 'connecting' || !isBleSupported}>Connect Device</button>
                  <button onClick={startSimulation} className="btn btn-outline" title="No hardware yet? Stream synthetic vitals to try the full pipeline.">
                    Simulate Device (Demo)
                  </button>
                </>
              ) : simulating ? (
                <button onClick={stopSimulation} className="btn btn-outline" style={{ color: 'var(--error)', borderColor: 'var(--error)' }}>Stop Simulation</button>
              ) : (
                <button onClick={handleBleDisconnect} className="btn btn-outline" style={{ color: 'var(--error)', borderColor: 'var(--error)' }}>Disconnect</button>
              )}
              {bleSamples.length > 0 && (
                <button onClick={handleBleExport} className="btn btn-outline" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Download size={14} /> Export CSV
                </button>
              )}
            </div>
          </div>

          {!isBleSupported && !simulating && bleStatus === 'idle' && (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '-1rem' }}>
              Web Bluetooth isn't supported in this browser (use Chrome/Edge for real hardware) — "Simulate Device" works everywhere and doesn't need it.
            </p>
          )}

          {/* Live metric tiles */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1rem' }}>
            {BLE_SIGNALS.map((sig) => {
              const val = (bleLatest as any)[sig.key];
              const active = bleStatus === 'connected' && val !== undefined;
              return (
                <div key={sig.key} className="glass-panel" style={{ padding: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                  <Circle size={10} fill={active ? sig.color : '#d1d5db'} color="transparent" />
                  <div>
                    <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{sig.label}</p>
                    <p style={{ fontWeight: 700, fontSize: '1.4rem', color: active ? sig.color : 'var(--text-muted)' }}>
                      {active ? val.toFixed(1) : '—'}
                      <span style={{ fontSize: '0.72rem', fontWeight: 400, color: 'var(--text-muted)' }}> {sig.unit}</span>
                    </p>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Save session */}
          {bleSamples.length > 0 && (
            <div className="glass-panel" style={{ padding: '1.1rem', display: 'flex', gap: '0.75rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: 200 }}>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Save recording as session</label>
                <input value={bleSessionName} onChange={(e) => setBleSessionName(e.target.value)} placeholder="e.g. ble_session_01"
                  style={{ width: '100%', padding: '0.6rem 0.75rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontSize: '0.9rem' }} />
              </div>
              <button onClick={handleBleSave} className="btn btn-primary" disabled={bleSaving || !bleSessionName.trim()}>
                {bleSaving ? 'Saving…' : 'Save to Portal'}
              </button>
            </div>
          )}

          {/* Event log */}
          <div className="glass-panel" style={{ padding: '1.1rem' }}>
            <p style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.6rem' }}>Event log</p>
            {bleLog.length === 0
              ? <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No events yet.</p>
              : bleLog.map((l, i) => <p key={i} style={{ fontFamily: 'monospace', fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.8 }}>{l}</p>)
            }
          </div>
        </div>
      )}
    </div>
  );
}

function DevRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-muted)' }}>{icon}{label}</span>
      <strong style={{ fontWeight: 600 }}>{value}</strong>
    </div>
  );
}
