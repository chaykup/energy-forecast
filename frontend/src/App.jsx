import { useState } from 'react'
import { useModelMetricsV2 } from './hooks/useModelData'
import MetricsTable from './components/MetricsTable'
import MetricBarChart from './components/MetricBarChart'
import PredictionOverlay from './components/PredictionOverlay'

export default function App() {
  const [market, setMarket] = useState('CAISO')
  const [view, setView] = useState({ node: null, regime: null })

  const { metrics, modelRegistry, nodes, loading, error } = useModelMetricsV2(market, view)

  if (error) return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center">
      <div className="text-red-400 bg-red-950/40 border border-red-800 rounded-xl px-6 py-4">
        Error: {error}
      </div>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">

      {/* ── Header ── */}
      <header className="border-b border-gray-800 px-8 py-5 flex items-center gap-8 flex-shrink-0">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white m-0">
            Energy LMP Forecast
          </h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Regime-conditional hybrid ML vs. TimeGPT · Hourly electricity price forecasting
          </p>
        </div>

        <div className="flex items-center gap-1 ml-auto bg-gray-900 border border-gray-700 rounded-lg p-1">
          {['CAISO', 'ERCOT'].map(m => (
            <button
              key={m}
              onClick={() => { setMarket(m); setView({ node: null, regime: null }) }}
              className={`px-5 py-1.5 rounded-md text-sm font-semibold transition-all ${
                market === m
                  ? 'bg-blue-600 text-white shadow'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </header>

      {/* ── Content ── */}
      <main className="flex-1 flex flex-col gap-4 p-6 min-w-0">
        {loading ? (
          <div className="flex-1 flex items-center justify-center text-gray-500 animate-pulse">
            Loading metrics…
          </div>
        ) : (
          <>
            {/* Row 1: Leaderboard table + Bar chart side by side */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <MetricsTable
                  metrics={metrics}
                  registry={modelRegistry}
                  view={view}
                  nodes={nodes}
                  onViewChange={setView}
                />
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <MetricBarChart metrics={metrics} registry={modelRegistry} />
              </div>
            </div>

            {/* Row 2: Prediction overlay — full width */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
              <PredictionOverlay market={market} registry={modelRegistry} />
            </div>
          </>
        )}
      </main>
    </div>
  )
}
