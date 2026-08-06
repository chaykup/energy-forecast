import { useState, useEffect, useMemo } from 'react'
import { supabase } from '../lib/supabase'

/**
 * Fetch model metrics for a given market.
 * Returns { metrics, modelRegistry, loading, error }
 */
export function useModelMetrics(market) {
  const [metrics, setMetrics] = useState([])
  const [modelRegistry, setModelRegistry] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function fetch() {
      setLoading(true)
      try {
        const [metricsRes, registryRes] = await Promise.all([
          supabase
            .from('model_metrics')
            .select('*')
            .eq('market', market)
            .order('mae', { ascending: true }),
          supabase
            .from('model_registry')
            .select('*')
            .order('sort_order'),
        ])

        if (metricsRes.error) throw metricsRes.error
        if (registryRes.error) throw registryRes.error

        setMetrics(metricsRes.data)
        setModelRegistry(registryRes.data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetch()
  }, [market])

  return { metrics, modelRegistry, loading, error }
}

/**
 * Fetch v2 model metrics with support for overall/node/regime views.
 * Loads all rows for the latest run_date in one query, then filters client-side.
 *
 * @param {string} market - 'CAISO' or 'ERCOT'
 * @param {object} view
 * @param {string|null} view.node   - filter to a specific node, or null
 * @param {number|null} view.regime - filter to a specific regime (0/1/2), or null for overall
 * @returns {{ metrics, modelRegistry, allRows, nodes, loading, error }}
 */
export function useModelMetricsV2(market, view = { node: null, regime: null }) {
  const [allRows, setAllRows] = useState([])
  const [modelRegistry, setModelRegistry] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function fetchAll() {
      setLoading(true)
      try {
        const [metricsRes, registryRes] = await Promise.all([
          supabase
            .from('model_metrics')
            .select('*')
            .eq('market', market)
            .order('run_date', { ascending: false }),
          supabase
            .from('model_registry')
            .select('*')
            .order('sort_order'),
        ])

        if (metricsRes.error) throw metricsRes.error
        if (registryRes.error) throw registryRes.error

        // De-duplicate: keep only the latest row per (model_name, node, regime).
        // This handles multiple uploads with different run_dates cleanly —
        // rows are already ordered descending by run_date, so first-seen wins.
        const seen = new Set()
        const latest = (metricsRes.data || []).filter(row => {
          const key = `${row.model_name}|${row.node}|${row.regime}`
          if (seen.has(key)) return false
          seen.add(key)
          return true
        })

        setAllRows(latest)
        setModelRegistry(registryRes.data || [])
      } catch (err) {
        setError(err?.message || String(err))
      } finally {
        setLoading(false)
      }
    }
    fetchAll()
  }, [market])

  // Client-side filtering based on view selection
  // regime === -1 means "overall aggregate" in the DB (sentinel for NULL avoidance)
  const metrics = useMemo(() => {
    if (view.regime !== null) {
      // Regime view: rows where regime = 0/1/2 and node IS NULL
      return allRows.filter(r => r.regime === view.regime && r.node === null)
    }
    if (view.node !== null) {
      // Node view: rows where node = X and regime = -1 (overall)
      return allRows.filter(r => r.node === view.node && r.regime === -1)
    }
    // Overall view: regime = -1 (overall sentinel) and node IS NULL
    return allRows.filter(r => r.regime === -1 && r.node === null)
  }, [allRows, view.node, view.regime])

  // Derive available nodes from allRows
  const nodes = useMemo(() => {
    const nodeSet = new Set(
      allRows
        .filter(r => r.node !== null && r.regime === -1)
        .map(r => r.node)
    )
    return Array.from(nodeSet).sort()
  }, [allRows])

  return { metrics, modelRegistry, allRows, nodes, loading, error }
}

/**
 * Fetch predictions for overlay chart.
 * Filters to a specific model + market, with optional date range.
 */
export function usePredictions(market, modelName, startHour, endHour) {
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!modelName) return

    async function fetch() {
      setLoading(true)
      let query = supabase
        .from('predictions')
        .select('hour, actual_lmp, predicted_lmp, regime, node')
        .eq('market', market)
        .eq('model_name', modelName)
        .order('hour')

      if (startHour) query = query.gte('hour', startHour)
      if (endHour) query = query.lte('hour', endHour)

      // Supabase returns max 1000 rows by default — paginate
      const allRows = []
      let offset = 0
      const PAGE_SIZE = 1000

      while (true) {
        const { data, error } = await query.range(offset, offset + PAGE_SIZE - 1)
        if (error) break
        allRows.push(...data)
        if (data.length < PAGE_SIZE) break
        offset += PAGE_SIZE
      }

      setPredictions(allRows)
      setLoading(false)
    }
    fetch()
  }, [market, modelName, startHour, endHour])

  return { predictions, loading }
}