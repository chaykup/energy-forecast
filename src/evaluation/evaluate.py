"""
Phase 4b: Extended Metrics Evaluation

Reads existing prediction parquets — does NOT retrain any models.
Computes: wMAPE, skill scores vs naive, arbitrage capture %,
          per-node breakdowns, per-regime breakdowns.
Outputs:  data/results/{MARKET}/model_comparison.json

Usage:
    python -m src.evaluation.evaluate --market CAISO
    python -m src.evaluation.evaluate --market ERCOT
    python -m src.evaluation.evaluate --market both
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    arbitrage_capture_pct,
    directional_accuracy,
    mae,
    rmse,
    skill_score,
    wmape,
)
from src.utils.config import PROCESSED_DIR, RESULTS_DIR

PARQUET_FILES = {
    "xgb_only":          "xgb_only_predictions.parquet",
    "hmm_xgb":           "hmm_xgb_predictions.parquet",
    "hybrid_full":       "hybrid_full_predictions.parquet",
    "timegpt_zero_shot": "timegpt_zero_shot_predictions.parquet",
    "timegpt_finetuned": "timegpt_finetuned_predictions.parquet",
}

# Val split end — test period starts the next hour
VAL_END = "2024-07-31"
# Train split end — used for naive baseline HOD lookup
TRAIN_END = "2024-03-31"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_features(market: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{market.lower()}_features.parquet"
    df = pd.read_parquet(path)
    df["hour"] = pd.to_datetime(df["hour"])
    return df


def _build_node_lookup(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns test-period rows with (timestamp, actual, Location) columns.
    Used to attach a Location label to each prediction row by matching
    on (timestamp, actual_lmp) — much more reliable than positional join
    because the model parquets are not consistently sorted.
    """
    test = features_df[features_df["hour"] > VAL_END][["hour", "Location", "LMP"]].copy()
    test = test.rename(columns={"hour": "timestamp", "LMP": "actual"})
    return test


def _attach_location(df: pd.DataFrame, node_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Location onto a prediction DataFrame via (timestamp, actual).
    When two nodes share the exact same LMP at the same timestamp (rare, ~9
    timestamps across the test period), the first matched location is kept.
    """
    indexed = df.reset_index()
    merged = indexed.merge(node_lookup, on=["timestamp", "actual"], how="left")
    # Drop duplicates from LMP collisions (keep first match per original row)
    merged = merged.drop_duplicates(subset="index").drop(columns="index").reset_index(drop=True)
    return merged


def _build_regime_map(market: str) -> dict:
    """
    timestamp → regime_state lookup from hmm_xgb_predictions.parquet.
    Regime is time-based (not node-based), so we deduplicate by timestamp
    and take the first occurrence.
    """
    path = RESULTS_DIR / market / "hmm_xgb_predictions.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.drop_duplicates("timestamp").set_index("timestamp")["regime_state"].to_dict()


def _compute_metrics_dict(
    actual: np.ndarray,
    predicted: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> dict:
    return {
        "mae":                   mae(actual, predicted),
        "rmse":                  rmse(actual, predicted),
        "wmape":                 wmape(actual, predicted),
        "directional_accuracy":  directional_accuracy(actual, predicted),
        "arbitrage_capture_pct": arbitrage_capture_pct(actual, predicted, timestamps),
        "n_samples":             int(len(actual)),
    }


def _inject_skill_scores(m: dict, naive_mae: float, naive_rmse: float) -> None:
    m["skill_score_mae"]  = skill_score(m["mae"],  naive_mae)
    m["skill_score_rmse"] = skill_score(m["rmse"], naive_rmse)


def _compute_for_df(df: pd.DataFrame, regime_map: dict) -> dict:
    """
    Compute overall, per-node, and per-regime metrics for a model DataFrame.
    Expects columns: timestamp, actual, predicted, Location (optional).
    Returns dict with keys: overall, by_node, by_regime.
    Skill scores are NOT set here — caller injects them afterward.
    """
    ts = pd.DatetimeIndex(df["timestamp"].values)
    overall = _compute_metrics_dict(df["actual"].values, df["predicted"].values, ts)

    by_node = {}
    if "Location" in df.columns:
        for node, node_df in df.groupby("Location"):
            if pd.isna(node):
                continue
            ts_n = pd.DatetimeIndex(node_df["timestamp"].values)
            by_node[node] = _compute_metrics_dict(
                node_df["actual"].values, node_df["predicted"].values, ts_n
            )

    # Regime breakdown: join via regime_map
    df = df.copy()
    df["regime_state"] = df["timestamp"].map(regime_map)
    by_regime = {}
    for rid in [0, 1, 2]:
        mask = df["regime_state"] == rid
        if mask.sum() < 10:
            continue
        rd = df[mask]
        ts_r = pd.DatetimeIndex(rd["timestamp"].values)
        m = _compute_metrics_dict(rd["actual"].values, rd["predicted"].values, ts_r)
        m["pct_of_total"] = float(mask.mean())
        by_regime[rid] = m

    return {"overall": overall, "by_node": by_node, "by_regime": by_regime}


# ---------------------------------------------------------------------------
# Naive baseline reconstruction
# ---------------------------------------------------------------------------

def reconstruct_naive_predictions(market: str, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build naive baseline predictions for the test period.
    Strategy: predict the historical mean LMP for each (Location, hour_of_day)
    using the training split (up to TRAIN_END), then apply to the test period.

    For locations absent from training data (e.g. ERCOT's HB_HOUSTON which has
    a data gap in the training window), falls back to the cross-location HOD mean.

    Returns DataFrame with columns: [timestamp, actual, predicted, Location]
    """
    train = features_df[features_df["hour"] <= TRAIN_END].copy()
    test = features_df[features_df["hour"] > VAL_END].copy()

    train["hour_of_day"] = train["hour"].dt.hour

    # Per-location HOD mean
    hod_mean = (
        train.groupby(["Location", "hour_of_day"])["LMP"]
        .mean()
        .reset_index()
        .rename(columns={"LMP": "naive_pred"})
    )

    # Cross-location fallback HOD mean (used when a location has no training data)
    hod_fallback = (
        train.groupby("hour_of_day")["LMP"]
        .mean()
        .reset_index()
        .rename(columns={"LMP": "fallback_pred"})
    )

    test = test.copy()
    test["hour_of_day"] = test["hour"].dt.hour
    test = test.merge(hod_mean, on=["Location", "hour_of_day"], how="left")
    test = test.merge(hod_fallback, on="hour_of_day", how="left")
    # Fill missing per-location predictions with cross-location fallback
    test["naive_pred"] = test["naive_pred"].fillna(test["fallback_pred"])

    return pd.DataFrame({
        "timestamp": test["hour"].values,
        "actual":    test["LMP"].values,
        "predicted": test["naive_pred"].values,
        "Location":  test["Location"].values,
    })


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate_market(market: str) -> None:
    print(f"\n{'='*60}")
    print(f"Evaluating {market}")
    print(f"{'='*60}")

    features_df = _load_features(market)
    node_lookup  = _build_node_lookup(features_df)
    regime_map   = _build_regime_map(market)

    all_results: dict[str, dict] = {}

    # --- Naive baseline (must be first for skill score computation) ---
    print("  naive_baseline ... ", end="", flush=True)
    naive_df = reconstruct_naive_predictions(market, features_df)
    all_results["naive_baseline"] = _compute_for_df(naive_df, regime_map)
    print(f"mae={all_results['naive_baseline']['overall']['mae']:.3f}")

    # Store naive reference metrics for skill score injection
    naive_overall = all_results["naive_baseline"]["overall"]
    naive_node    = all_results["naive_baseline"]["by_node"]
    naive_regime  = all_results["naive_baseline"]["by_regime"]

    # --- All other models ---
    for model_name, parquet_file in PARQUET_FILES.items():
        print(f"  {model_name} ... ", end="", flush=True)
        path = RESULTS_DIR / market / parquet_file
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Attach Location via LMP-value merge
        df = _attach_location(df, node_lookup)

        all_results[model_name] = _compute_for_df(df, regime_map)
        print(f"mae={all_results[model_name]['overall']['mae']:.3f}")

    # --- Inject skill scores ---
    for model_name, result in all_results.items():
        overall = result["overall"]
        _inject_skill_scores(
            overall,
            naive_overall["mae"],
            naive_overall["rmse"],
        )

        for node, m in result["by_node"].items():
            naive_m = naive_node.get(node, naive_overall)
            _inject_skill_scores(m, naive_m["mae"], naive_m["rmse"])

        for rid, m in result["by_regime"].items():
            naive_m = naive_regime.get(rid, naive_overall)
            _inject_skill_scores(m, naive_m["mae"], naive_m["rmse"])

    # --- Build v2 JSON ---
    leaderboard = [
        {
            "model":                  name,
            "market":                 market,
            "mae":                    r["overall"]["mae"],
            "rmse":                   r["overall"]["rmse"],
            "wmape":                  r["overall"]["wmape"],
            "skill_score_mae":        r["overall"]["skill_score_mae"],
            "skill_score_rmse":       r["overall"]["skill_score_rmse"],
            "arbitrage_capture_pct":  r["overall"]["arbitrage_capture_pct"],
            "directional_accuracy":   r["overall"]["directional_accuracy"],
            "n_samples":              r["overall"]["n_samples"],
        }
        for name, r in all_results.items()
    ]
    leaderboard.sort(key=lambda x: x["mae"])

    v2 = {
        "version":    "v2",
        "market":     market,
        "leaderboard": leaderboard,
        "by_node":    {name: r["by_node"]   for name, r in all_results.items()},
        "by_regime":  {name: {str(k): v for k, v in r["by_regime"].items()}
                       for name, r in all_results.items()},
    }

    def _sanitize(obj):
        """Recursively replace NaN/inf floats with None for valid JSON."""
        if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    out_path = RESULTS_DIR / market / "model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(_sanitize(v2), f, indent=2)
    print(f"\nSaved: {out_path}")

    # Quick sanity print
    print("\nLeaderboard (MAE):")
    for entry in leaderboard:
        arb = entry["arbitrage_capture_pct"]
        arb_str = f"{arb:.1f}%" if arb is not None and not (isinstance(arb, float) and np.isnan(arb)) else "n/a"
        print(f"  {entry['model']:30s} mae={entry['mae']:.4f}  arb_cap={arb_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute v2 extended metrics from saved parquets")
    parser.add_argument(
        "--market",
        choices=["CAISO", "ERCOT", "both"],
        default="both",
        help="Market to evaluate",
    )
    args = parser.parse_args()

    markets = ["CAISO", "ERCOT"] if args.market == "both" else [args.market]
    for market in markets:
        evaluate_market(market)


if __name__ == "__main__":
    main()
