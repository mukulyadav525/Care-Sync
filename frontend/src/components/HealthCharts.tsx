"use client";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

// NOTE: this previously used Tailwind utility classes (h-80, bg-[#111827],
// text-white) but this project doesn't have Tailwind installed, so none of
// them applied — the chart <div>s had no height, so ResponsiveContainer
// (height="100%" of a 0px parent) rendered nothing, leaving just the bare
// <h2> titles visible. Switched to inline styles with explicit pixel
// heights, matching the working pattern in components/trends/WeeklyTrendChart.

const CHART_HEIGHT = 260;

function ChartCard({
  title,
  data,
  dataKey,
  color,
  unit,
}: {
  title: string;
  data: any[];
  dataKey: string;
  color: string;
  unit: string;
}) {
  const hasData = data.some((d) => d?.[dataKey] !== undefined && d?.[dataKey] !== null);

  return (
    <div style={{ background: "#111827", borderRadius: 16, padding: 24 }}>
      <h2 style={{ color: "white", fontSize: 20, marginBottom: 12 }}>{title}</h2>
      {!hasData ? (
        <p style={{ color: "#9CA3AF", fontSize: "0.85rem" }}>No data available.</p>
      ) : (
        <div style={{ height: CHART_HEIGHT }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid stroke="#333" />
              <XAxis dataKey="hour" stroke="#9CA3AF" tick={{ fill: "#9CA3AF", fontSize: 12 }} />
              <YAxis stroke="#9CA3AF" tick={{ fill: "#9CA3AF", fontSize: 12 }} unit={unit} />
              <Tooltip contentStyle={{ background: "#1f2937", border: "none", borderRadius: 8 }} labelStyle={{ color: "#e5e7eb" }} />
              <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export default function HealthCharts({ data }: { data: any[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <ChartCard title="Heart Rate" data={data} dataKey="heart_rate" color="#00ff99" unit=" bpm" />
      <ChartCard title="Stress Score" data={data} dataKey="stress_score" color="#ff5555" unit="" />
      <ChartCard title="RMSSD" data={data} dataKey="rmssd" color="#3399ff" unit=" ms" />
    </div>
  );
}
