"use client";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

type HourPoint = {
  hour: number;
  value: number;
};

type DayData = {
  day: string;
  data: HourPoint[];
};

interface Props {
  title: string;
  unit: string;
  data: DayData[];
}

export default function WeeklyTrendChart({
  title,
  unit,
  data,
}: Props) {
  return (
    <div
      style={{
        background: "#111827",
        borderRadius: 16,
        padding: 24,
      }}
    >
      <h2
        style={{
          color: "white",
          marginBottom: 25,
          fontSize: 22,
        }}
      >
        {title}
      </h2>

      {data.map((day, index) => (
        <div
          key={day.day}
          style={{
            display: "flex",
            alignItems: "center",
            marginBottom: 20,
          }}
        >
          <div
            style={{
              width: 70,
              color: "#9CA3AF",
              fontWeight: 600,
            }}
          >
            {day.day}
          </div>

          <div
            style={{
              flex: 1,
              height: 80,
            }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={day.data}>
                <CartesianGrid
                  stroke="#374151"
                  strokeDasharray="3 3"
                />

                <XAxis
                  dataKey="hour"
                  ticks={[
                    0, 4, 8, 12, 16, 20, 24,
                  ]}
                  stroke="#6B7280"
                />

                <YAxis
                  width={35}
                  stroke="#6B7280"
                />

                <Tooltip />

                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ))}

      <div
        style={{
          textAlign: "center",
          color: "#9CA3AF",
          marginTop: 10,
        }}
      >
        Hour of Day (24 Hours)
      </div>
    </div>
  );
}