"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ShieldCheck, CheckCircle } from 'lucide-react';
import api from '@/lib/api';

type Step = 'personal' | 'health' | 'lifestyle' | 'review' | 'done';

interface FormData {
  // Personal
  age: string;
  gender: string;
  height: string;
  weight: string;
  // Health
  respiratory_conditions: string[];
  cardiovascular_conditions: string[];
  cardiovascular_symptoms: string[];
  metabolic_conditions: string[];
  mental_health_conditions: string[];
  stress_level: string;
  // Lifestyle
  lifestyle_factors: string[];
  sleep_hours: string;
  sleep_disorders: string[];
  last_medical_checkup: string;
  health_concerns: string;
  // Consent
  consent_data: boolean;
  consent_participate: boolean;
  consent_withdraw: boolean;
}

const initial: FormData = {
  age: '', gender: '', height: '', weight: '',
  respiratory_conditions: [], cardiovascular_conditions: [], cardiovascular_symptoms: [],
  metabolic_conditions: [], mental_health_conditions: [], stress_level: '',
  lifestyle_factors: [], sleep_hours: '', sleep_disorders: [],
  last_medical_checkup: '', health_concerns: '',
  consent_data: false, consent_participate: false, consent_withdraw: false,
};

function MultiCheck({ label, options, value, onChange }: { label: string; options: string[]; value: string[]; onChange: (v: string[]) => void }) {
  const toggle = (opt: string) =>
    onChange(value.includes(opt) ? value.filter((x) => x !== opt) : [...value, opt]);
  return (
    <div style={{ marginBottom: '1.25rem' }}>
      <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-muted)' }}>{label}</p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
        {options.map((opt) => (
          <label key={opt} style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer', fontSize: '0.85rem', padding: '0.35rem 0.75rem', border: `1px solid ${value.includes(opt) ? 'var(--primary)' : 'var(--border)'}`, borderRadius: '999px', background: value.includes(opt) ? 'rgba(16,185,129,0.1)' : 'transparent', color: value.includes(opt) ? 'var(--primary)' : 'inherit', transition: 'all 0.1s' }}>
            <input type="checkbox" checked={value.includes(opt)} onChange={() => toggle(opt)} style={{ display: 'none' }} />
            {opt}
          </label>
        ))}
      </div>
    </div>
  );
}

const STEPS: Step[] = ['personal', 'health', 'lifestyle', 'review', 'done'];
const STEP_LABELS = ['Personal Info', 'Health History', 'Lifestyle', 'Review & Consent'];

export default function ConsentFormPage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>('personal');
  const [form, setForm] = useState<FormData>(initial);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const set = (key: keyof FormData, val: any) => setForm((f) => ({ ...f, [key]: val }));

  const stepIdx = STEPS.indexOf(step);

  const handleSubmit = async () => {
    setSubmitting(true); setError('');
    try {
      await api.post('/files/form-submit/', {
        age: parseInt(form.age),
        gender: form.gender,
        height: parseFloat(form.height),
        weight: parseFloat(form.weight),
        respiratory_conditions: form.respiratory_conditions.join(', '),
        cardiovascular_conditions: form.cardiovascular_conditions.join(', '),
        cardiovascular_symptoms: form.cardiovascular_symptoms.join(', '),
        metabolic_conditions: form.metabolic_conditions.join(', '),
        mental_health_conditions: form.mental_health_conditions.join(', '),
        stress_level: form.stress_level,
        lifestyle_factors: form.lifestyle_factors.join(', '),
        sleep_hours: form.sleep_hours,
        sleep_disorders: form.sleep_disorders.join(', '),
        last_medical_checkup: form.last_medical_checkup,
        health_concerns: form.health_concerns,
      });
      setStep('done');
    } catch (err: any) {
      setError(err.response?.data?.error || 'Submission failed. Please try again.');
    } finally { setSubmitting(false); }
  };

  const fieldStyle = { width: '100%', padding: '0.65rem 0.85rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', background: 'var(--surface)', color: 'inherit', fontSize: '0.9rem' };
  const labelStyle = { fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', display: 'block' as const, marginBottom: 4 };

  if (step === 'done') {
    return (
      <div className="container" style={{ maxWidth: 560, margin: '0 auto', padding: '5rem 1rem', textAlign: 'center' }}>
        <CheckCircle size={56} color="var(--success)" style={{ marginBottom: '1.5rem' }} />
        <h2 style={{ marginBottom: '0.75rem' }}>Consent submitted</h2>
        <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>Thank you for completing the health consent form. Your responses have been securely recorded.</p>
        <button onClick={() => router.push('/dashboard')} className="btn btn-primary">Return to Dashboard</button>
      </div>
    );
  }

  return (
    <div className="container animate-fade-in" style={{ maxWidth: 700, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header */}
      <div>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}><ShieldCheck size={24} color="var(--primary)" /> Health Consent Form</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
          This information helps the research team understand your health background. It is stored securely and only accessible to authorised clinicians.
        </p>
      </div>

      {/* Step indicator */}
      <div style={{ display: 'flex', gap: 0 }}>
        {STEP_LABELS.map((label, i) => (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem' }}>
            <div style={{ width: '100%', height: 4, borderRadius: 2, background: i <= stepIdx ? 'var(--primary)' : 'var(--border)', transition: 'background 0.2s' }} />
            <span style={{ fontSize: '0.72rem', color: i <= stepIdx ? 'var(--primary)' : 'var(--text-muted)', fontWeight: i === stepIdx ? 700 : 400, textAlign: 'center' }}>{label}</span>
          </div>
        ))}
      </div>

      <div className="glass-panel" style={{ padding: '2rem' }}>

        {/* Step 1: Personal */}
        {step === 'personal' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <h3 style={{ marginBottom: '0.25rem' }}>Personal Information</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={labelStyle}>Age *</label>
                <input type="number" min="1" max="120" style={fieldStyle} value={form.age} onChange={(e) => set('age', e.target.value)} placeholder="e.g. 28" required />
              </div>
              <div>
                <label style={labelStyle}>Gender *</label>
                <select style={fieldStyle} value={form.gender} onChange={(e) => set('gender', e.target.value)} required>
                  <option value="">— select —</option>
                  {['Male', 'Female', 'Non-binary', 'Prefer not to say'].map((g) => <option key={g}>{g}</option>)}
                </select>
              </div>
              <div>
                <label style={labelStyle}>Height (cm) *</label>
                <input type="number" step="0.1" style={fieldStyle} value={form.height} onChange={(e) => set('height', e.target.value)} placeholder="e.g. 172" />
              </div>
              <div>
                <label style={labelStyle}>Weight (kg) *</label>
                <input type="number" step="0.1" style={fieldStyle} value={form.weight} onChange={(e) => set('weight', e.target.value)} placeholder="e.g. 68" />
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Health history */}
        {step === 'health' && (
          <div>
            <h3 style={{ marginBottom: '1.25rem' }}>Health History</h3>
            <MultiCheck label="Respiratory conditions" options={['Asthma', 'COPD', 'Sleep apnea', 'None']} value={form.respiratory_conditions} onChange={(v) => set('respiratory_conditions', v)} />
            <MultiCheck label="Cardiovascular conditions" options={['Hypertension', 'Arrhythmia', 'Heart failure', 'Coronary artery disease', 'None']} value={form.cardiovascular_conditions} onChange={(v) => set('cardiovascular_conditions', v)} />
            <MultiCheck label="Cardiovascular symptoms" options={['Chest pain', 'Palpitations', 'Shortness of breath', 'Dizziness', 'None']} value={form.cardiovascular_symptoms} onChange={(v) => set('cardiovascular_symptoms', v)} />
            <MultiCheck label="Metabolic conditions" options={['Type 1 diabetes', 'Type 2 diabetes', 'Thyroid disorder', 'Obesity', 'None']} value={form.metabolic_conditions} onChange={(v) => set('metabolic_conditions', v)} />
            <MultiCheck label="Mental health conditions" options={['Anxiety', 'Depression', 'PTSD', 'None']} value={form.mental_health_conditions} onChange={(v) => set('mental_health_conditions', v)} />
            <div>
              <label style={labelStyle}>Typical stress level *</label>
              <select style={fieldStyle} value={form.stress_level} onChange={(e) => set('stress_level', e.target.value)}>
                <option value="">— select —</option>
                {['Low', 'Moderate', 'High', 'Very high'].map((s) => <option key={s}>{s}</option>)}
              </select>
            </div>
          </div>
        )}

        {/* Step 3: Lifestyle */}
        {step === 'lifestyle' && (
          <div>
            <h3 style={{ marginBottom: '1.25rem' }}>Lifestyle</h3>
            <MultiCheck label="Lifestyle factors" options={['Smoker', 'Former smoker', 'Alcohol use', 'Regular exercise', 'Sedentary', 'None']} value={form.lifestyle_factors} onChange={(v) => set('lifestyle_factors', v)} />
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={labelStyle}>Average sleep per night *</label>
              <select style={fieldStyle} value={form.sleep_hours} onChange={(e) => set('sleep_hours', e.target.value)}>
                <option value="">— select —</option>
                {['< 5 hours', '5–6 hours', '6–7 hours', '7–8 hours', '> 8 hours'].map((s) => <option key={s}>{s}</option>)}
              </select>
            </div>
            <MultiCheck label="Sleep disorders" options={['Insomnia', 'Sleep apnea', 'Restless legs', 'None']} value={form.sleep_disorders} onChange={(v) => set('sleep_disorders', v)} />
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={labelStyle}>Last medical check-up *</label>
              <select style={fieldStyle} value={form.last_medical_checkup} onChange={(e) => set('last_medical_checkup', e.target.value)}>
                <option value="">— select —</option>
                {['< 6 months', '6–12 months', '1–2 years', '> 2 years', 'Never'].map((s) => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div>
              <label style={labelStyle}>Any other health concerns?</label>
              <textarea rows={3} style={{ ...fieldStyle, resize: 'vertical' }} value={form.health_concerns} onChange={(e) => set('health_concerns', e.target.value)} placeholder="Optional…" />
            </div>
          </div>
        )}

        {/* Step 4: Review & Consent */}
        {step === 'review' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <h3>Review & Consent</h3>
            <div style={{ padding: '1rem', background: 'var(--info-bg)', border: '1px solid var(--info-border)', borderRadius: 'var(--radius-md)', fontSize: '0.88rem', lineHeight: 1.7 }}>
              <p><strong>Summary</strong></p>
              <p>Age: {form.age} · Gender: {form.gender} · Height: {form.height} cm · Weight: {form.weight} kg</p>
              <p>Stress: {form.stress_level || '—'} · Sleep: {form.sleep_hours || '—'}</p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {[
                ['consent_data', 'I consent to my physiological data being collected and stored securely for research purposes.'],
                ['consent_participate', 'I understand I am voluntarily participating in this study and my data will be anonymised.'],
                ['consent_withdraw', 'I understand I can withdraw my consent and request data deletion at any time.'],
              ].map(([key, text]) => (
                <label key={key} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem', cursor: 'pointer', fontSize: '0.88rem', lineHeight: 1.5 }}>
                  <input type="checkbox" checked={(form as any)[key]} onChange={(e) => set(key as keyof FormData, e.target.checked)}
                    style={{ marginTop: 3, accentColor: 'var(--primary)', width: 16, height: 16, flexShrink: 0 }} />
                  {text}
                </label>
              ))}
            </div>
            {error && <p style={{ color: 'var(--error)', fontSize: '0.85rem' }}>{error}</p>}
          </div>
        )}

        {/* Navigation */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '2rem' }}>
          {stepIdx > 0 ? (
            <button onClick={() => setStep(STEPS[stepIdx - 1])} className="btn btn-outline">← Back</button>
          ) : <div />}
          {stepIdx < STEPS.indexOf('review') ? (
            <button onClick={() => setStep(STEPS[stepIdx + 1])} className="btn btn-primary">Next →</button>
          ) : (
            <button onClick={handleSubmit} className="btn btn-primary"
              disabled={submitting || !form.consent_data || !form.consent_participate || !form.consent_withdraw}>
              {submitting ? 'Submitting…' : 'Submit Consent'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
