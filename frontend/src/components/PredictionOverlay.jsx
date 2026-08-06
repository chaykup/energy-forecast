import { useState, useMemo, useCallback } from 'react'
import { usePredictions } from '../hooks/useModelData'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid,
} from 'recharts'

function downsample(arr, maxPoints = 1000) {
  if (arr.length <= maxPoints) return arr
  const stride = Math.ceil(arr.length / maxPoints)
  return arr.filter((_, i) => i % stride === 0)
}

export default function PredictionOverlay({ market, registry }) {
  const [selectedModel, setSelectedModel] = useState('hmm_xgb_lstm')

  const { predictions, loading } = usePredictions(market, selectedModel)

  const chartData = useMemo(() => {
    if (!predictions.length) return []
    const firstNode = predictions[0].node
    const filtered = firstNode ? predictions.filter(r => r.node === firstNode) : predictions
    return downsample(filtered, 1000)
  }, [predictions])

  const tickFormatter = useCallback(
    v => new Date(v).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
    []
  )
  const labelFormatter = useCallback(v => new Date(v).toLocaleString(), [])

  const lineColor = registry.find(r => r.model_name === selectedModel)?.color || '#ef4444'

  return (
    <div>
      {/* Header row */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-widest">
          Actual vs. Predicted LMP
        </h2>
        <div className="flex flex-wrap gap-1.5">
          {registry.map(r => (
            <button
              key={r.model_name}
              onClick={() => setSelectedModel(r.model_name)}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all border ${
                selectedModel === r.model_name
                  ? 'border-current bg-gray-800'
                  : 'border-gray-700 text-gray-500 hover:text-gray-300 hover:border-gray-600'
              }`}
              style={{ color: selectedModel === r.model_name ? r.color : undefined }}
            >
              {r.display_name}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-72 text-gray-600 animate-pulse text-sm">
          Loading predictions…
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: 8 }}>
            <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="hour"
              stroke="#374151"
              tick={{ fontSize: 11, fill: '#6b7280' }}
              tickLine={false}
              tickFormatter={tickFormatter}
              minTickGap={60}
            />
            <YAxis
              stroke="#374151"
              tick={{ fontSize: 11, fill: '#6b7280' }}
              tickLine={false}
              axisLine={false}
              label={{ value: '$/MWh', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
              labelStyle={{ color: '#e5e7eb', fontWeight: 600 }}
              labelFormatter={labelFormatter}
              isAnimationActive={false}
            />
            <Legend
              wrapperStyle={{ fontSize: 12, color: '#9ca3af', paddingTop: 12 }}
            />
            <Line
              type="monotone" dataKey="actual_lmp" name="Actual"
              stroke="#6b7280" dot={false} strokeWidth={1.5}
              isAnimationActive={false}
            />
            <Line
              type="monotone" dataKey="predicted_lmp" name="Predicted"
              stroke={lineColor} dot={false} strokeWidth={2}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
