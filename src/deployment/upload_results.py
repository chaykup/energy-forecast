"""
Upload model results to Supabase.

Uploads v2 metrics (wMAPE, skill scores, arbitrage capture, per-node/regime rows)
and raw prediction parquets in one pass.

Usage:
    python -m src.deployment.upload_results --market CAISO
    python -m src.deployment.upload_results --market ERCOT
    python -m src.deployment.upload_results --market both
    python -m src.deployment.upload_results --market both --sample-rate 6  # ~4h resolution
    python -m src.deployment.upload_results --market both --skip-predictions
    python -m src.deployment.upload_results --market both --skip-metrics

Reads:  data/results/{market}/model_comparison_v2.json
        data/results/{market}/*_predictions.parquet
Writes: model_metrics table  (wmape, skill_score_mae/rmse, arbitrage_capture_pct, regime, node)
        predictions table
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.deployment.supabase_client import get_supabase_client

RESULTS_DIR = Path("data/results")
BATCH_SIZE = 500

MODEL_KEY_MAP = {
    "naive_baseline":    "naive_baseline",
    "xgb_only":          "xgb_only",
    "hmm_xgb":           "hmm_xgb",
    "hybrid_full":       "hmm_xgb_lstm",
    "timegpt_zero_shot": "timegpt_zero_shot",
    "timegpt_finetuned": "timegpt_finetuned",
}

PREDICTION_FILES = {
    "xgb_only_predictions.parquet":          "xgb_only",
    "hmm_xgb_predictions.parquet":           "hmm_xgb",
    "hybrid_full_predictions.parquet":       "hmm_xgb_lstm",
    "timegpt_zero_shot_predictions.parquet": "timegpt_zero_shot",
    "timegpt_finetuned_predictions.parquet": "timegpt_finetuned",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _build_metric_row(
    market: str,
    model_key: str,
    m: dict,
    run_date: str,
    node: str | None = None,
    regime: int = -1,
) -> dict:
    return {
        "market":                market,
        "model_name":            MODEL_KEY_MAP.get(model_key, model_key),
        "node":                  node,
        "regime":                regime,
        "run_date":              run_date,
        "mae":                   m.get("mae"),
        "rmse":                  m.get("rmse"),
        "wmape":                 m.get("wmape"),
        "skill_score_mae":       m.get("skill_score_mae"),
        "skill_score_rmse":      m.get("skill_score_rmse"),
        "arbitrage_capture_pct": m.get("arbitrage_capture_pct"),
        "directional_accuracy":  m.get("directional_accuracy"),
        "peak_hour_mae":         None,
        "spike_recall":          None,
        "metadata": json.dumps({
            "n_samples":    m.get("n_samples"),
            "pct_of_total": m.get("pct_of_total"),
        }),
    }


def upload_metrics(market: str, supabase) -> None:
    json_path = RESULTS_DIR / market / "model_comparison.json"
    if not json_path.exists():
        print(f"  ⚠ {json_path} not found — run evaluate.py first")
        return

    with open(json_path) as f:
        v2 = json.load(f)

    run_date = str(date.today())
    rows = []

    for entry in v2.get("leaderboard", []):
        rows.append(_build_metric_row(market, entry["model"], entry, run_date))

    for model_key, node_dict in v2.get("by_node", {}).items():
        for node, m in node_dict.items():
            rows.append(_build_metric_row(market, model_key, m, run_date, node=node))

    for model_key, regime_dict in v2.get("by_regime", {}).items():
        for regime_str, m in regime_dict.items():
            rows.append(_build_metric_row(
                market, model_key, m, run_date, regime=int(regime_str)
            ))

    for i in range(0, len(rows), BATCH_SIZE):
        supabase.table("model_metrics").upsert(
            rows[i:i + BATCH_SIZE], ignore_duplicates=True
        ).execute()

    print(
        f"  ✓ Uploaded {len(rows)} metric rows for {market} "
        f"({len(v2.get('leaderboard', []))} overall, "
        f"{sum(len(v) for v in v2.get('by_node', {}).values())} node, "
        f"{sum(len(v) for v in v2.get('by_regime', {}).values())} regime)"
    )


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def upload_predictions(market: str, supabase, sample_rate: int = 1) -> None:
    market_dir = RESULTS_DIR / market

    for filename, model_name in PREDICTION_FILES.items():
        path = market_dir / filename
        if not path.exists():
            print(f"  ⚠ {path} not found, skipping")
            continue

        df = pd.read_parquet(path)

        col_map = {}
        for candidate in ["hour", "ds", "timestamp"]:
            if candidate in df.columns:
                col_map[candidate] = "hour"
                break
        for candidate in ["actual_lmp", "LMP", "actual", "y"]:
            if candidate in df.columns:
                col_map[candidate] = "actual_lmp"
                break
        for candidate in ["predicted_lmp", "predicted", "TimeGPT", "prediction"]:
            if candidate in df.columns:
                col_map[candidate] = "predicted_lmp"
                break
        if "regime_state" in df.columns:
            col_map["regime_state"] = "regime"

        df = df.rename(columns=col_map)
        if sample_rate > 1:
            df = df.iloc[::sample_rate]

        rows = [
            {
                "market":        market,
                "model_name":    model_name,
                "node":          r.get("node") or r.get("Location") or r.get("unique_id"),
                "hour":          str(r["hour"]),
                "actual_lmp":    float(r["actual_lmp"]) if pd.notna(r.get("actual_lmp")) else None,
                "predicted_lmp": float(r["predicted_lmp"]) if pd.notna(r.get("predicted_lmp")) else None,
                "regime":        int(r["regime"]) if "regime" in df.columns and pd.notna(r.get("regime")) else None,
            }
            for _, r in df.iterrows()
        ]

        for i in range(0, len(rows), BATCH_SIZE):
            supabase.table("predictions").upsert(rows[i:i + BATCH_SIZE]).execute()

        print(f"  ✓ Uploaded {len(rows)} prediction rows for {model_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model results to Supabase")
    parser.add_argument("--market", required=True, choices=["CAISO", "ERCOT", "both"])
    parser.add_argument(
        "--sample-rate", type=int, default=1,
        help="Upload every Nth prediction row (default: 1 = all; 6 = ~4h resolution)",
    )
    parser.add_argument("--skip-metrics",     action="store_true", help="Skip metrics upload")
    parser.add_argument("--skip-predictions", action="store_true", help="Skip predictions upload")
    args = parser.parse_args()

    supabase = get_supabase_client()
    markets = ["CAISO", "ERCOT"] if args.market == "both" else [args.market]

    for market in markets:
        print(f"\n{'='*60}")
        print(f"  Uploading {market} to Supabase")
        print(f"{'='*60}")
        if not args.skip_metrics:
            upload_metrics(market, supabase)
        if not args.skip_predictions:
            upload_predictions(market, supabase, sample_rate=args.sample_rate)

    print("\n✓ All uploads complete")


if __name__ == "__main__":
    main()
