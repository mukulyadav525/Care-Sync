"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Users, Radio, FileText, CheckCircle, XCircle, ChevronRight } from 'lucide-react';
import api from '@/lib/api';

interface PatientRow {
  username: string;
  email: string;
  date_joined: string;
  session_count: number;
  last_session: string | null;
  consent: boolean;
}

export default function AdminPatientsPage() {
  const router = useRouter();
  const [patients, setPatients] = useState<PatientRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    api.get('/admin/patients/').then((r) => {
      setPatients(r.data.patients);
      setLoading(false);
    }).catch((err) => {
      if (err.response?.status === 401) router.push('/login');
      else if (err.response?.status === 403) setError('Admin access required.');
      else setError('Failed to load patients.');
      setLoading(false);
    });
  }, [router]);

  if (loading) return <div className="container" style={{ padding: '4rem', textAlign: 'center' }}>Loading…</div>;
  if (error) return <div className="container" style={{ padding: '4rem', textAlign: 'center', color: 'var(--error)' }}>{error}</div>;

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Users size={24} color="var(--primary)" /> Patients</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>{patients.length} registered participants</p>
      </div>

      {/* Summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
        {[
          { label: 'Total patients', value: patients.length, color: 'var(--primary)' },
          { label: 'Consent submitted', value: patients.filter((p) => p.consent).length, color: 'var(--success)' },
          { label: 'With sessions', value: patients.filter((p) => p.session_count > 0).length, color: '#0d9488' },
          { label: 'No consent yet', value: patients.filter((p) => !p.consent).length, color: 'var(--error)' },
        ].map((c) => (
          <div key={c.label} className="glass-panel" style={{ padding: '1.1rem' }}>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>{c.label}</p>
            <p style={{ fontWeight: 700, fontSize: '1.75rem', color: c.color }}>{c.value}</p>
          </div>
        ))}
      </div>

      {/* Patient table */}
      <div className="glass-panel" style={{ overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.88rem' }}>
          <thead>
            <tr style={{ background: 'var(--panel-bg-light)', borderBottom: '1px solid var(--border)' }}>
              <th style={{ padding: '0.85rem 1.25rem' }}>Username</th>
              <th style={{ padding: '0.85rem 1rem' }}>Email</th>
              <th style={{ padding: '0.85rem 1rem' }}>Joined</th>
              <th style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>Sessions</th>
              <th style={{ padding: '0.85rem 1rem' }}>Last session</th>
              <th style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>Consent</th>
              <th style={{ padding: '0.85rem 1rem' }} />
            </tr>
          </thead>
          <tbody>
            {patients.map((p) => (
              <tr key={p.username} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '0.85rem 1.25rem', fontWeight: 600 }}>{p.username}</td>
                <td style={{ padding: '0.85rem 1rem', color: 'var(--text-muted)' }}>{p.email}</td>
                <td style={{ padding: '0.85rem 1rem', color: 'var(--text-muted)' }}>
                  {new Date(p.date_joined).toLocaleDateString()}
                </td>
                <td style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', color: p.session_count > 0 ? 'var(--primary)' : 'var(--text-muted)', fontWeight: 600 }}>
                    <Radio size={13} />{p.session_count}
                  </span>
                </td>
                <td style={{ padding: '0.85rem 1rem', color: 'var(--text-muted)' }}>
                  {p.last_session ? new Date(p.last_session).toLocaleDateString() : '—'}
                </td>
                <td style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>
                  {p.consent
                    ? <CheckCircle size={17} color="var(--success)" />
                    : <XCircle size={17} color="var(--error)" />}
                </td>
                <td style={{ padding: '0.85rem 1rem' }}>
                  {p.session_count > 0 && (
                    <Link href={`/portal?owner=${p.username}`} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: 'var(--primary)', fontSize: '0.8rem', fontWeight: 600 }}>
                      View <ChevronRight size={14} />
                    </Link>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
