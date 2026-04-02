"""
Phase 5: Upload model results to Supabase.

Usage:
    python -m src.deployment.upload_results --market CAISO
    python -m src.deployment.upload_results --market both

Reads from:  data/results/{market}/model_comparison.json
             data/results/{market}/*_predictions.parquet
Writes to:   Supabase Postgres (model_metrics, predictions tables)
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from src.deployment.supabase_client import get_supabase_client

RESULTS_DIR = Path("data/results")

# Map JSON model names to DB model_name values
MODEL_KEY_MAP = {
    "naive_baseline": "naive_baseline",
    "timegpt_zero_shot": "timegpt_zero_shot",
    "timegpt_finetuned": "timegpt_finetuned",
    "xgb_only": "xgb_only",
    "hmm_xgb": "hmm_xgb",
    "hybrid_full": "hmm_xgb_lstm",
}

# Map prediction filenames to model names
PREDICTION_FILES = {
    "xgb_only_predictions.parquet": "xgb_only",
    "hmm_xgb_predictions.parquet": "hmm_xgb",
    "hybrid_full_predictions.parquet": "hmm_xgb_lstm",
    "timegpt_zero_shot_predictions.parquet": "timegpt_zero_shot",
    "timegpt_finetuned_predictions.parquet": "timegpt_finetuned",
}


def upload_metrics(market: str, supabase):
    """Upload model_comparison.json metrics to model_metrics table."""
    json_path = RESULTS_DIR / market / "model_comparison.json"
    if not json_path.exists():
        print(f"  ⚠ {json_path} not found, skipping metrics upload")
        return

    with open(json_path) as f:
        comparison = json.load(f)

    leaderboard = comparison.get("leaderboard", [])
    if not leaderboard:
        print(f"  ⚠ No leaderboard found in {json_path}")
        return

    rows = []
    for entry in leaderboard:
        raw_key = entry.get("model", "")
        model_name = MODEL_KEY_MAP.get(raw_key, raw_key)
        row = {
            "market": market,
            "model_name": model_name,
            "node": None,
            "mae": entry.get("mae"),
            "rmse": entry.get("rmse"),
            "mape": entry.get("mape"),           # NULL — not in your JSON
            "r2": entry.get("r2"),               # NULL — not in your JSON
            "directional_accuracy": entry.get("directional_accuracy"),
            "peak_hour_mae": entry.get("peak_hour_mae"),  # NULL
            "spike_recall": entry.get("spike_recall"),     # NULL
            "metadata": json.dumps({
                "median_ae": entry.get("median_ae"),
                "max_error": entry.get("max_error"),
                "n_samples": entry.get("n_samples"),
            }),
        }
        rows.append(row)

    result = supabase.table("model_metrics").upsert(rows).execute()
    print(f"  ✓ Uploaded {len(rows)} metric rows for {market}")

def upload_predictions(market: str, supabase, sample_rate: int = 1):
    """
    Upload prediction parquets to predictions table.

    Args:
        sample_rate: Upload every Nth row. Use 1 for full data,
                     6 for ~4h resolution (keeps DB small on free tier).
    """
    market_dir = RESULTS_DIR / market

    for filename, model_name in PREDICTION_FILES.items():
        path = market_dir / filename
        if not path.exists():
            print(f"  ⚠ {path} not found, skipping")
            continue

        df = pd.read_parquet(path)

        # Standardize column names — your parquets may vary
        col_map = {}
        # Timestamp column → hour
        for candidate in ["hour", "ds", "timestamp"]:
            if candidate in df.columns:
                col_map[candidate] = "hour"
                break

        # Actual LMP column
        for candidate in ["actual_lmp", "LMP", "actual", "y"]:
            if candidate in df.columns:
                col_map[candidate] = "actual_lmp"
                break

        # Predicted LMP column
        for candidate in ["predicted_lmp", "predicted", "TimeGPT", "prediction"]:
            if candidate in df.columns:
                col_map[candidate] = "predicted_lmp"
                break

        # Regime column
        if "regime_state" in df.columns:
            col_map["regime_state"] = "regime"

        df = df.rename(columns=col_map)

        # Sample down if needed
        if sample_rate > 1:
            df = df.iloc[::sample_rate]

        rows = []
        for _, r in df.iterrows():
            row = {
                "market": market,
                "model_name": model_name,
                "node": r.get("node") or r.get("Location") or r.get("unique_id"),
                "hour": str(r["hour"]),
                "actual_lmp": float(r["actual_lmp"]) if pd.notna(r.get("actual_lmp")) else None,
                "predicted_lmp": float(r["predicted_lmp"]) if pd.notna(r.get("predicted_lmp")) else None,
                "regime": int(r["regime"]) if "regime" in df.columns and pd.notna(r.get("regime")) else None,
            }
            rows.append(row)

        # Supabase has a ~1000 row batch limit — chunk uploads
        BATCH_SIZE = 500
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            supabase.table("predictions").upsert(batch).execute()

        print(f"  ✓ Uploaded {len(rows)} prediction rows for {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Upload results to Supabase")
    parser.add_argument("--market", required=True, choices=["CAISO", "ERCOT", "both"])
    parser.add_argument("--sample-rate", type=int, default=1,
                        help="Upload every Nth prediction row (default: 1 = all)")
    args = parser.parse_args()

    supabase = get_supabase_client()
    markets = ["CAISO", "ERCOT"] if args.market == "both" else [args.market]

    for market in markets:
        print(f"\n{'=' * 60}")
        print(f"  Uploading {market} results to Supabase")
        print(f"{'=' * 60}")
        upload_metrics(market, supabase)
        upload_predictions(market, supabase, sample_rate=args.sample_rate)

    print("\n✓ All uploads complete")


if __name__ == "__main__":
    main()