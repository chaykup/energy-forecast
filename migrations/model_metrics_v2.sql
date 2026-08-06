-- Migration: model_metrics v2
-- Adds new metric columns, drops retired ones, and updates the unique constraint
-- to support per-regime and per-node rows alongside overall rows.
--
-- Run this in the Supabase SQL editor BEFORE running upload_results_v2.py.

-- 1. Add new metric columns
ALTER TABLE model_metrics
  ADD COLUMN IF NOT EXISTS wmape                FLOAT,
  ADD COLUMN IF NOT EXISTS skill_score_mae      FLOAT,
  ADD COLUMN IF NOT EXISTS skill_score_rmse     FLOAT,
  ADD COLUMN IF NOT EXISTS arbitrage_capture_pct FLOAT;

-- 2. Add regime column
--    -1 = overall aggregate (all regimes combined)
--     0, 1, 2 = HMM regime segmentation
ALTER TABLE model_metrics
  ADD COLUMN IF NOT EXISTS regime INTEGER NOT NULL DEFAULT -1;

COMMENT ON COLUMN model_metrics.regime IS
  '-1 = overall (all regimes aggregated), 0/1/2 = HMM regime segmentation';

-- 3. Drop retired columns
ALTER TABLE model_metrics
  DROP COLUMN IF EXISTS mape,
  DROP COLUMN IF EXISTS r2;

-- 4. Replace unique constraint to include regime
--    Postgres UNIQUE constraints don't support expressions, so we use a
--    unique index instead. COALESCE(node, '') treats NULL node as '' so
--    two "overall" rows (node=NULL) can't conflict with each other.
ALTER TABLE model_metrics
  DROP CONSTRAINT IF EXISTS model_metrics_market_model_name_node_run_date_key;

DROP INDEX IF EXISTS model_metrics_unique_v2;

CREATE UNIQUE INDEX model_metrics_unique_v2
  ON model_metrics (market, model_name, COALESCE(node, ''), run_date, regime);
