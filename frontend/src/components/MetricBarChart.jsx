import { useState, useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const METRICS = [
  { key: 'mae',                   label: 'MAE' },
  { key: 'rmse',                  label: 'RMSE' },
  { key: 'wmape',                 label: 'wMAPE' },
  { key: 'skill_score_mae',       label: 'Skill' },
  { key: 'arbitrage_capture_pct', label: 'Arb %' },
]

export default function MetricBarChart({ metrics, registry }) {
  const [activeMetric, setActiveMetric] = useState('mae')

  const data = useMemo(() => metrics.map(m => {
    const reg = registry.find(r => r.model_name === m.model_name) || {}
    let value = m[activeMetric]
    if (activeMetric === 'wmape' && value != null) value = value * 100
    return { name: reg.display_name || m.model_name, value, color: reg.color || '#666' }
  }).filter(d => d.value != null), [metrics, registry, activeMetric])

  const activeLabel = METRICS.find(m => m.key === activeMetric)?.label ?? activeMetric

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-widest">
          By Metric
        </h2>
      </div>

      {/* Metric selector */}
      <div className="flex flex-wrap gap-1.5 mb-5">
        {METRICS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveMetric(key)}
            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
              activeMetric === key
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-700'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 0, bottom: 0 }}>
          <XAxis
            type="number"
            stroke="#444"
            tick={{ fontSize: 10, fill: '#6b7280' }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="#444"
            tick={{ fontSize: 11, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={false}
            width={110}
          />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
            contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
            labelStyle={{ color: '#e5e7eb', fontWeight: 600 }}
            formatter={v => [`${v?.toFixed(4)}`, activeLabel]}
            isAnimationActive={false}
          />
          <Bar dataKey="value" radius={[0, 5, 5, 0]} maxBarSize={24}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.color} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
