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

export default function HealthCharts({
  data,
}: {
  data: any[];
}) {
  return (
    <div className="space-y-8">

      <div className="h-80 bg-[#111827] rounded-xl p-4">
        <h2 className="text-white text-xl mb-3">
          Heart Rate
        </h2>

        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="#333" />

            <XAxis dataKey="hour" />

            <YAxis />

            <Tooltip />

            <Line
              type="monotone"
              dataKey="heart_rate"
              stroke="#00ff99"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="h-80 bg-[#111827] rounded-xl p-4">
        <h2 className="text-white text-xl mb-3">
          Stress Score
        </h2>

        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="#333" />

            <XAxis dataKey="hour" />

            <YAxis />

            <Tooltip />

            <Line
              dataKey="stress_score"
              stroke="#ff5555"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="h-80 bg-[#111827] rounded-xl p-4">
        <h2 className="text-white text-xl mb-3">
          RMSSD
        </h2>

        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="#333" />

            <XAxis dataKey="hour" />

            <YAxis />

            <Tooltip />

            <Line
              dataKey="rmssd"
              stroke="#3399ff"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

    </div>
  );
}