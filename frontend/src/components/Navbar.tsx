"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter, usePathname } from 'next/navigation';
import { Activity, LogOut, User, FileText, BarChart2, Radio, Cpu, Bell, Sun, Moon, TrendingUp, Users } from 'lucide-react';
import api from '@/lib/api';
import { useTheme } from '@/components/ThemeProvider';

export default function Navbar() {
  const router = useRouter();
  const pathname = usePathname();
  const { theme, toggle: toggleTheme } = useTheme();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUsername] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('accessToken');
    const userStr = localStorage.getItem('user');
    if (token && userStr) {
      setIsAuthenticated(true);
      try {
        const user = JSON.parse(userStr);
        setUsername(user.username);
        setIsAdmin(!!user.is_superuser);
      } catch (e) {}
    } else {
      setIsAuthenticated(false);
      setIsAdmin(false);
    }
  }, [pathname]);

  const handleLogout = async () => {
    try {
      const refresh = localStorage.getItem('refreshToken');
      if (refresh) {
        await api.post('/auth/logout/', { refresh });
      }
    } catch (e) {
      console.error('Logout error', e);
    } finally {
      localStorage.removeItem('accessToken');
      localStorage.removeItem('refreshToken');
      localStorage.removeItem('user');
      setIsAuthenticated(false);
      router.push('/login');
    }
  };

  // Hide Navbar on the root page and auth pages to match the requested design
  if (pathname === '/' || pathname === '/login' || pathname === '/signup') {
    return null;
  }

  return (
    <nav style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '1rem 2rem',
      backgroundColor: 'var(--surface-color)',
      borderBottom: '1px solid var(--border-color)',
      position: 'sticky',
      top: 0,
      zIndex: 100,
    }}>
      <Link href="/dashboard" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.25rem', fontWeight: 700, color: 'var(--primary-color)' }}>
        <Activity size={28} />
        Care-Sync
      </Link>

      <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
        <Link href="/dashboard" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: pathname === '/dashboard' ? 'var(--primary-color)' : 'var(--text-primary)', fontWeight: 600 }}>
          <BarChart2 size={18} /> Dashboard
        </Link>
        <Link href="/portal" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: (pathname.startsWith('/portal') && !pathname.startsWith('/portal/trends') && !pathname.startsWith('/portal/compare')) ? 'var(--primary-color)' : 'var(--text-primary)', fontWeight: 600 }}>
          <Radio size={18} /> Signals
        </Link>
        <Link href="/portal/trends" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: pathname.startsWith('/portal/trends') ? 'var(--primary-color)' : 'var(--text-primary)', fontWeight: 600 }}>
          <TrendingUp size={18} /> Trends
        </Link>
        {isAdmin && (
          <Link href="/admin/patients" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: pathname.startsWith('/admin') ? 'var(--primary-color)' : 'var(--text-primary)', fontWeight: 600 }}>
            <Users size={18} /> Patients
          </Link>
        )}
        <Link href="/files" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: pathname.startsWith('/files') ? 'var(--primary-color)' : 'var(--text-primary)', fontWeight: 600 }}>
          <FileText size={18} /> Files
        </Link>
        <Link href="/devices" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: pathname.startsWith('/devices') ? 'var(--primary-color)' : 'var(--text-primary)', fontWeight: 600 }}>
          <Cpu size={18} /> Devices
        </Link>
        <Link href="/alerts" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: pathname.startsWith('/alerts') ? 'var(--primary-color)' : 'var(--text-primary)', fontWeight: 600 }}>
          <Bell size={18} /> Alerts
        </Link>
        <div style={{ width: '1px', height: '24px', background: 'var(--border-color)' }}></div>
        <button onClick={toggleTheme} title="Toggle theme" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', padding: '0.25rem' }}>
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </button>
        <Link href="/settings" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: pathname.startsWith('/settings') ? 'var(--primary-color)' : 'var(--text-secondary)', fontWeight: 600 }} title="Settings">
          <User size={18} /> {username}
        </Link>
        <button onClick={handleLogout} className="btn" style={{ padding: '0.5rem 1rem', border: '1px solid var(--border-color)', color: 'var(--text-primary)', backgroundColor: 'transparent' }}>
          <LogOut size={16} /> Sign out
        </button>
      </div>
    </nav>
  );
}
