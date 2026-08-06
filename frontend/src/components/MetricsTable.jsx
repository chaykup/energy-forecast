import { useState } from 'react'

const METRIC_COLS = [
  { key: 'mae',                   label: 'MAE',         lower_better: true,  fmt: v => v?.toFixed(3) ?? '—' },
  { key: 'rmse',                  label: 'RMSE',        lower_better: true,  fmt: v => v?.toFixed(3) ?? '—' },
  { key: 'wmape',                 label: 'wMAPE',       lower_better: true,  fmt: v => v != null ? `${(v * 100).toFixed(2)}%` : '—' },
  { key: 'skill_score_mae',       label: 'Skill (MAE)', lower_better: false, fmt: v => v?.toFixed(4) ?? '—' },
  { key: 'arbitrage_capture_pct', label: 'Arb. Cap %',  lower_better: false, fmt: v => v != null ? `${v.toFixed(1)}%` : '—' },
]

function shortNode(node) {
  return node.replace('TH_', '').replace('_GEN-APND', '').replace('HB_', '')
}

const REGIME_LABELS = { 0: 'Low Vol', 1: 'Normal', 2: 'High Vol' }

export default function MetricsTable({ metrics, registry, view, nodes, onViewChange }) {
  const [sortCol, setSortCol] = useState('mae')
  const [sortAsc, setSortAsc] = useState(true)

  const handleSort = (col) => {
    if (sortCol === col) setSortAsc(a => !a)
    else { setSortCol(col); setSortAsc(true) }
  }

  const rows = metrics.map(m => ({
    ...m,
    ...(registry.find(r => r.model_name === m.model_name) || {}),
  })).sort((a, b) => {
    const va = a[sortCol] ?? Infinity
    const vb = b[sortCol] ?? Infinity
    return sortAsc ? va - vb : vb - va
  })

  const best = {}
  METRIC_COLS.forEach(({ key, lower_better }) => {
    const vals = rows.map(r => r[key]).filter(v => v != null)
    best[key] = lower_better ? Math.min(...vals) : Math.max(...vals)
  })

  const isOverall = view.node === null && view.regime === null

  return (
    <div>
      {/* Section header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-widest">
          Model Leaderboard
        </h2>

        {/* View selector */}
        <div className="flex items-center gap-1.5 flex-wrap justify-end">
          <button
            onClick={() => onViewChange({ node: null, regime: null })}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
              isOverall
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-700'
            }`}
          >
            Overall
          </button>

          {nodes.length > 0 && nodes.map(n => (
            <button
              key={n}
              onClick={() => onViewChange({ node: n, regime: null })}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                view.node === n
                  ? 'bg-violet-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-700'
              }`}
            >
              {shortNode(n)}
            </button>
          ))}

          {[0, 1, 2].map(r => (
            <button
              key={r}
              onClick={() => onViewChange({ node: null, regime: r })}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                view.regime === r
                  ? 'bg-amber-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-700'
              }`}
            >
              {REGIME_LABELS[r]}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto -mx-1">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700/60">
              <th className="text-left py-2.5 px-3 text-xs font-medium text-gray-500 uppercase tracking-wider">
                Model
              </th>
              {METRIC_COLS.map(({ key, label }) => (
                <th
                  key={key}
                  onClick={() => handleSort(key)}
                  className="text-right py-2.5 px-3 text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer select-none hover:text-gray-300 transition-colors"
                >
                  {label}
                  <span className="ml-1 opacity-60">
                    {sortCol === key ? (sortAsc ? '↑' : '↓') : ''}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/60">
            {rows.map((row, i) => (
              <tr
                key={row.model_name}
                className="hover:bg-gray-800/40 transition-colors group"
              >
                <td className="py-3 px-3">
                  <div className="flex items-center gap-2.5">
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: row.color || '#666', boxShadow: `0 0 6px ${row.color || '#666'}60` }}
                    />
                    <span className="font-medium text-gray-200 whitespace-nowrap">
                      {row.display_name || row.model_name}
                    </span>
                    {i === 0 && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-900/60 text-green-400 border border-green-800/60 font-medium">
                        BEST
                      </span>
                    )}
                  </div>
                </td>
                {METRIC_COLS.map(({ key, fmt }) => {
                  const isBest = row[key] != null && row[key] === best[key]
                  return (
                    <td
                      key={key}
                      className={`text-right py-3 px-3 font-mono text-sm tabular-nums ${
                        isBest ? 'text-green-400 font-semibold' : 'text-gray-400'
                      }`}
                    >
                      {fmt(row[key])}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
