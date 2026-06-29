"use client";

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Brush } from 'recharts';
import api from '@/lib/api';
import { Activity, HeartPulse } from 'lucide-react';

export default function PPGVisualization() {
  const params = useParams();
  const owner = params.owner as string;
  const filename = params.filename as string;
  
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedDay, setSelectedDay] = useState(1);

  useEffect(() => {
    fetchData(selectedDay);
  }, [selectedDay, owner, filename]);

  const fetchData = async (day: number) => {
    setLoading(true);
    try {
      const res = await api.get(`/visualization/local-ppg/${filename}/?day=${day}`);
      if (res.data.error) throw new Error(res.data.error);
      setData(res.data);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to fetch PPG data');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !data) return <div className="container" style={{ padding: '4rem', textAlign: 'center' }}>Loading high-frequency data...</div>;
  if (error) return <div className="container" style={{ padding: '4rem', textAlign: 'center', color: 'var(--error)' }}>{error}</div>;

  const currentDayData = data?.days?.find((d: any) => d.day === selectedDay);

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>PPG Signal Analysis</h2>
          <p style={{ color: 'var(--text-muted)' }}>File: {filename} | Owner: {owner}</p>
        </div>
        
        {/* Day Selector */}
        {data?.total_days > 1 && (
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <span style={{ fontWeight: 500 }}>Select Day:</span>
            <select 
              value={selectedDay} 
              onChange={(e) => setSelectedDay(parseInt(e.target.value))}
              className="input-field"
              style={{ width: 'auto', padding: '0.5rem' }}
            >
              {Array.from({ length: data.total_days }, (_, i) => (
                <option key={i+1} value={i+1}>Day {i+1}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ background: 'rgba(67, 97, 238, 0.1)', padding: '1rem', borderRadius: '50%' }}>
            <HeartPulse color="var(--primary)" />
          </div>
          <div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Heart Rate (Avg)</p>
            <h3>{currentDayData?.stats?.hr || 'N/A'} bpm</h3>
          </div>
        </div>
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ background: 'rgba(114, 9, 183, 0.1)', padding: '1rem', borderRadius: '50%' }}>
            <Activity color="var(--secondary)" />
          </div>
          <div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>HRV (RMSSD)</p>
            <h3>{currentDayData?.stats?.hrv || 'N/A'} ms</h3>
          </div>
        </div>
      </div>

      {/* Interactive Chart */}
      <div className="glass-panel" style={{ padding: '2rem', height: '600px', display: 'flex', flexDirection: 'column' }}>
        <h3 style={{ marginBottom: '1rem' }}>PPG Waveform (Day {selectedDay})</h3>
        {loading ? (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Loading Day {selectedDay}...</div>
        ) : (
          <div style={{ flex: 1, width: '100%', height: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={currentDayData?.series || []} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis 
                  dataKey="time" 
                  tick={{ fill: 'var(--text-muted)' }} 
                  axisLine={{ stroke: 'var(--border)' }}
                  minTickGap={50}
                />
                <YAxis 
                  domain={['auto', 'auto']}
                  tick={{ fill: 'var(--text-muted)' }}
                  axisLine={{ stroke: 'var(--border)' }}
                />
                <Tooltip 
                  contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '8px' }}
                  itemStyle={{ color: 'var(--primary)' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="ppg" 
                  stroke="var(--primary)" 
                  strokeWidth={2} 
                  dot={false}
                  activeDot={{ r: 6, fill: 'var(--secondary)' }}
                />
                <Brush dataKey="time" height={40} stroke="var(--primary)" fill="var(--glass-bg)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
