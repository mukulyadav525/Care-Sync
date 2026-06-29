"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Heart, ShieldCheck } from 'lucide-react';
import api from '@/lib/api';

type Mode = 'signin' | 'signup';
type Step = 'credentials' | 'otp';

export default function AuthPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>('signin');
  const [step, setStep] = useState<Step>('credentials');

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [signupData, setSignupData] = useState({ username: '', email: '', password1: '', password2: '' });

  const [otp, setOtp] = useState('');
  const [maskedEmail, setMaskedEmail] = useState('');
  const [devOtp, setDevOtp] = useState('');   // dev-only convenience

  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const persistSession = (data: any) => {
    localStorage.setItem('accessToken', data.access);
    localStorage.setItem('refreshToken', data.refresh);
    localStorage.setItem('user', JSON.stringify(data.user));
    router.push('/dashboard');
  };

  const resetTo = (m: Mode) => {
    setMode(m); setStep('credentials'); setError(''); setOtp(''); setDevOtp('');
  };

  // --- Sign in (step 1: password) -> sends 2FA code
  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault(); setError(''); setLoading(true);
    try {
      const res = await api.post('/auth/login/', { username, password });
      if (res.data.otp_required) {
        setMaskedEmail(res.data.email || '');
        setDevOtp(res.data.dev_otp || '');
        setStep('otp');
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Invalid credentials.');
    } finally { setLoading(false); }
  };

  // --- Sign up (step 1) -> sends verification code
  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    if (signupData.password1 !== signupData.password2) { setError('Passwords do not match.'); return; }
    setLoading(true);
    try {
      const res = await api.post('/auth/signup/', signupData);
      setMaskedEmail(res.data.email || signupData.email);
      setDevOtp(res.data.dev_otp || '');
      setStep('otp');
    } catch (err: any) {
      setError(err.response?.data?.error || 'Signup failed.');
    } finally { setLoading(false); }
  };

  // --- Step 2: verify the emailed code (login or signup)
  const handleVerify = async (e: React.FormEvent) => {
    e.preventDefault(); setError(''); setLoading(true);
    try {
      const res = mode === 'signin'
        ? await api.post('/auth/login/verify/', { username, otp })
        : await api.post('/auth/verify-otp/', { email: signupData.email, otp });
      persistSession(res.data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Invalid code.');
    } finally { setLoading(false); }
  };

  const colStyle = { display: 'flex', flexDirection: 'column' as const, flex: 1 };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem', backgroundColor: 'var(--bg-color)' }}>
      <div style={{ display: 'flex', width: '100%', maxWidth: '1200px', minHeight: '720px', gap: '2rem' }}>

        {/* Left panel */}
        <div style={{ flex: 1, backgroundColor: 'var(--panel-bg-light)', borderRadius: 'var(--radius-lg)', padding: '3rem', display: 'flex', flexDirection: 'column', border: '1px solid var(--panel-border)' }}>
          <div>
            <div style={{ width: '48px', height: '48px', backgroundColor: 'var(--primary-color)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.5rem' }}>
              <Heart color="white" fill="transparent" size={24} />
            </div>
            <p style={{ color: 'var(--primary-color)', fontWeight: 700, letterSpacing: '0.1em', fontSize: '0.875rem', marginBottom: '0.5rem' }}>CARE-SYNC</p>
            <h1 style={{ fontSize: '2.75rem', lineHeight: 1.1, marginBottom: '1.5rem' }}>Protected medical<br />workspace</h1>
          </div>
          <p style={{ fontSize: '1.15rem', color: 'var(--text-secondary)', marginBottom: '2.5rem', maxWidth: '85%' }}>
            A privacy-first health companion with two-factor sign-in, patient-owned records and advanced signal processing.
          </p>
          <div style={{ display: 'flex', gap: '2rem', marginBottom: 'auto' }}>
            <div style={{ flex: 1 }}>
              <h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>Two-Factor Security</h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', lineHeight: 1.6 }}>Every sign-in is confirmed with a one-time code sent to your email.</p>
            </div>
            <div style={{ flex: 1 }}>
              <h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>Patient-Owned Records</h4>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', lineHeight: 1.6 }}>Medical records and vital signals stay scoped to the signed-in user.</p>
            </div>
          </div>
          <div style={{ backgroundColor: 'var(--info-bg)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--info-border)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
              <ShieldCheck size={16} /><span style={{ fontSize: '0.875rem', fontWeight: 600 }}>Your data is protected</span>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.5 }}>
              Passwords are validated for strength and never stored in plain text. Codes expire in 10 minutes.
            </p>
          </div>
        </div>

        {/* Right panel */}
        <div style={{ flex: 1, backgroundColor: 'var(--surface-color)', borderRadius: 'var(--radius-lg)', padding: '4rem', display: 'flex', flexDirection: 'column', border: '1px solid var(--border-color)' }}>

          {step === 'credentials' && (
            <div style={{ display: 'flex', backgroundColor: 'var(--btn-toggle-bg)', borderRadius: 'var(--radius-md)', padding: '0.25rem', marginBottom: '2.5rem' }}>
              {(['signin', 'signup'] as Mode[]).map((m) => (
                <button key={m} onClick={() => resetTo(m)} style={{
                  flex: 1, padding: '0.75rem', borderRadius: 'calc(var(--radius-md) - 0.25rem)', border: 'none', fontWeight: 600, cursor: 'pointer', transition: 'var(--transition-fast)',
                  backgroundColor: mode === m ? 'var(--btn-toggle-active)' : 'transparent',
                  color: mode === m ? 'var(--btn-toggle-text-active)' : 'var(--btn-toggle-text)',
                }}>
                  {m === 'signin' ? 'Sign in' : 'Create account'}
                </button>
              ))}
            </div>
          )}

          <h2 style={{ fontSize: '1.85rem', marginBottom: '0.5rem' }}>
            {step === 'otp' ? 'Verify it’s you' : (mode === 'signin' ? 'Welcome back' : 'Create an account')}
          </h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>
            {step === 'otp'
              ? `Enter the 6-digit code sent to ${maskedEmail || 'your email'}.`
              : (mode === 'signin' ? 'Sign in to access your medical records.' : 'Sign up to create your secure workspace.')}
          </p>

          {error && (
            <div style={{ padding: '0.75rem', backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', borderRadius: 'var(--radius-sm)', marginBottom: '1.5rem', fontSize: '0.875rem', fontWeight: 500 }}>
              {error}
            </div>
          )}

          {devOtp && step === 'otp' && (
            <div style={{ padding: '0.6rem 0.75rem', backgroundColor: 'var(--panel-bg-light)', color: 'var(--primary-hover)', borderRadius: 'var(--radius-sm)', marginBottom: '1.25rem', fontSize: '0.8rem', border: '1px solid var(--panel-border)' }}>
              Dev mode — your code is <strong>{devOtp}</strong>
            </div>
          )}

          {/* Credentials: sign in */}
          {step === 'credentials' && mode === 'signin' && (
            <form onSubmit={handleSignIn} style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
              <div className="input-group">
                <label className="input-label">Username</label>
                <input type="text" className="input-field" placeholder="Enter your username" value={username} onChange={e => setUsername(e.target.value)} required />
              </div>
              <div className="input-group">
                <label className="input-label">Password</label>
                <input type="password" className="input-field" placeholder="Enter your password" value={password} onChange={e => setPassword(e.target.value)} required />
              </div>
              <button type="submit" className="btn btn-dark" style={{ marginTop: 'auto', width: '100%', padding: '1rem' }} disabled={loading}>
                {loading ? 'Sending code…' : 'Continue'}
              </button>
            </form>
          )}

          {/* Credentials: sign up */}
          {step === 'credentials' && mode === 'signup' && (
            <form onSubmit={handleSignUp} style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <div className="input-group" style={colStyle}>
                  <label className="input-label">Username</label>
                  <input type="text" className="input-field" value={signupData.username} onChange={e => setSignupData({ ...signupData, username: e.target.value })} required />
                </div>
                <div className="input-group" style={colStyle}>
                  <label className="input-label">Email</label>
                  <input type="email" className="input-field" placeholder="you@example.com" value={signupData.email} onChange={e => setSignupData({ ...signupData, email: e.target.value })} required />
                </div>
              </div>
              <div className="input-group">
                <label className="input-label">Password</label>
                <input type="password" className="input-field" placeholder="At least 8 chars, not all numbers" value={signupData.password1} onChange={e => setSignupData({ ...signupData, password1: e.target.value })} required />
              </div>
              <div className="input-group">
                <label className="input-label">Confirm Password</label>
                <input type="password" className="input-field" value={signupData.password2} onChange={e => setSignupData({ ...signupData, password2: e.target.value })} required />
              </div>
              <button type="submit" className="btn btn-dark" style={{ marginTop: 'auto', width: '100%', padding: '1rem' }} disabled={loading}>
                {loading ? 'Sending code…' : 'Create secure account'}
              </button>
            </form>
          )}

          {/* OTP step (shared) */}
          {step === 'otp' && (
            <form onSubmit={handleVerify} style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
              <div className="input-group">
                <label className="input-label">Verification code</label>
                <input type="text" inputMode="numeric" className="input-field" value={otp} onChange={e => setOtp(e.target.value.replace(/\D/g, ''))} maxLength={6} required
                  style={{ textAlign: 'center', letterSpacing: '0.5rem', fontSize: '1.5rem', padding: '1.25rem' }} />
              </div>
              <button type="submit" className="btn btn-dark" style={{ width: '100%', padding: '1rem' }} disabled={loading || otp.length < 6}>
                {loading ? 'Verifying…' : 'Verify & continue'}
              </button>
              <button type="button" onClick={() => { setStep('credentials'); setError(''); setOtp(''); }} style={{ marginTop: '1rem', background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.85rem' }}>
                ← Back
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
