"""
Phase 4: Training & Validation Orchestrator

Trains all 6 model variants on the same temporal splits for fair comparison:
  1. Naive Baseline (no training — rule-based)
  2. TimeGPT Zero-Shot (no training — API call)
  3. TimeGPT Fine-Tuned (API call with finetune_steps)
  4. XGBoost Only (single model, no regimes, no LSTM)
  5. HMM + XGBoost (regime-conditional XGBoost, no LSTM)
  6. HMM + XGBoost + LSTM (full custom pipeline)

Usage:
    python -m src.training.train_all_models --market CAISO
    python -m src.training.train_all_models --market ERCOT
    python -m src.training.train_all_models --market both
    python -m src.training.train_all_models --market CAISO --skip-timegpt
"""

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.hmm_regime import RegimeDetector
from src.models.xgboost_model import RegimeXGBoost
from src.models.lstm_model import RegimeLSTM
from src.models.hybrid_pipeline import HybridPipeline
from src.models.timegpt_baseline import TimeGPTBaseline
from src.evaluation.model_comparator import ModelComparator, ModelResult
from src.evaluation.backtest import backtest_strategy, naive_baseline_pnl
from src.training.walk_forward import WalkForwardValidator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPLIT_CONFIG = {
    "CAISO": {
        "train_end": "2024-03-31",
        "val_end": "2024-07-31",
    },
    "ERCOT": {
        "train_end": "2024-03-31",
        "val_end": "2024-07-31",
    },
}

# XGBoost params for the single-model variant (uses eval_set + early stopping)
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "early_stopping_rounds": 50,
}

# XGBoost params for per-regime models (no eval_set, so no early stopping)
REGIME_XGB_PARAMS = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}

LSTM_PARAMS = {
    "seq_len": 24,
    "hidden_size": 64,
    "num_layers": 2,
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 32,
}

HMM_N_REGIMES = 3

DATA_DIR = Path("data")
MODEL_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"


# ---------------------------------------------------------------------------
# Data Loading & Splitting
# ---------------------------------------------------------------------------

def load_feature_matrix(market: str) -> pd.DataFrame:
    """Load the processed feature matrix (output of feature_engineering.py)."""
    path = DATA_DIR / "processed" / f"{market.lower()}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. "
            f"Run feature_engineering.py first."
        )
    df = pd.read_parquet(path)

    df["hour"] = pd.to_datetime(df["hour"])
    df = df.sort_values("hour").reset_index(drop=True)

    if "Time" in df.columns:
        if df["Time"].dt.tz is not None:
            df["Time"] = df["Time"].dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            df["Time"] = pd.to_datetime(df["Time"])

    return df


def temporal_split(df: pd.DataFrame, market: str) -> tuple:
    """
    Split into train/val/test using temporal boundaries.
    Uses the "hour" column (naive UTC) for all comparisons.
    """
    config = SPLIT_CONFIG[market]
    train_end = pd.Timestamp(config["train_end"])
    val_end = pd.Timestamp(config["val_end"])

    train = df[df["hour"] <= train_end].copy()
    val = df[(df["hour"] > train_end) & (df["hour"] <= val_end)].copy()
    test = df[df["hour"] > val_end].copy()

    print(f"\n  Temporal split for {market}:")
    print(f"    Train: {train['hour'].min()} → {train['hour'].max()}  ({len(train):,} rows)")
    print(f"    Val:   {val['hour'].min()} → {val['hour'].max()}  ({len(val):,} rows)")
    print(f"    Test:  {test['hour'].min()} → {test['hour'].max()}  ({len(test):,} rows)")

    if len(train) == 0:
        raise ValueError("Training set is empty. Check SPLIT_CONFIG dates vs data range.")
    if len(val) == 0:
        raise ValueError("Validation set is empty. Check SPLIT_CONFIG dates vs data range.")
    if len(test) == 0:
        raise ValueError("Test set is empty. Check SPLIT_CONFIG dates vs data range.")

    return train, val, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns, excluding target and metadata."""
    return [c for c in df.select_dtypes(include=["number", "bool"]).columns
            if c not in ("LMP", "regime")]


# ---------------------------------------------------------------------------
# Model 4: XGBoost Only (no regimes, no LSTM)
# ---------------------------------------------------------------------------

class XGBoostOnlyModel:
    """Single XGBoost model — no regime detection, no LSTM residual correction."""

    def __init__(self, market: str):
        self.market = market
        self.model = None
        self.feature_cols = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        self.feature_cols = get_feature_cols(train_df)
        X_train = train_df[self.feature_cols]
        y_train = train_df["LMP"]
        X_val = val_df[self.feature_cols]
        y_val = val_df["LMP"]

        self.model = xgb.XGBRegressor(**XGB_PARAMS)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        print(f"    XGBoost-Only best iteration: {self.model.best_iteration}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.feature_cols])

    def save(self, path: str):
        joblib.dump({
            "model": self.model,
            "feature_cols": self.feature_cols,
            "market": self.market,
        }, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostOnlyModel":
        data = joblib.load(path)
        obj = cls(market=data["market"])
        obj.model = data["model"]
        obj.feature_cols = data["feature_cols"]
        return obj


# ---------------------------------------------------------------------------
# Model 5: HMM + XGBoost (no LSTM)
# ---------------------------------------------------------------------------

class HMMXGBoostModel:
    """Regime-conditional XGBoost without LSTM residual correction."""

    def __init__(self, market: str, n_regimes: int = HMM_N_REGIMES):
        self.market = market
        self.n_regimes = n_regimes
        self.regime_detector = RegimeDetector(n_regimes=n_regimes)
        self.xgb_models = {}

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        print("    Fitting HMM...")
        self.regime_detector.fit(train_df)
        X_hmm = self.regime_detector.prepare_observations(train_df)
        states = self.regime_detector.model.predict(X_hmm)

        train_aligned = train_df.iloc[-len(states):].copy()
        train_aligned["regime"] = states

        for rid in range(self.n_regimes):
            n = (states == rid).sum()
            label = self.regime_detector.regime_labels.get(rid, "?")
            print(f"      Regime {rid} ({label}): {n:,} samples")

        for rid in range(self.n_regimes):
            print(f"    Training XGBoost for regime {rid}...")
            model = RegimeXGBoost(regime_id=rid, market=self.market, **REGIME_XGB_PARAMS)
            model.fit(train_aligned, states)
            self.xgb_models[rid] = model

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (predictions, regime_states)."""
        X_hmm = self.regime_detector.prepare_observations(df)
        states = self.regime_detector.model.predict(X_hmm)
        df_aligned = df.iloc[-len(states):].copy()

        preds = np.zeros(len(df_aligned))
        for rid in range(self.n_regimes):
            mask = states == rid
            if mask.sum() > 0:
                regime_df = df_aligned[mask]
                preds[mask] = self.xgb_models[rid].predict(regime_df)

        return preds, states

    def save(self, base_path: str):
        self.regime_detector.save(f"{base_path}/{self.market}_hmm_xgb_only_hmm.joblib")
        for rid, model in self.xgb_models.items():
            model.save(f"{base_path}/{self.market}_hmm_xgb_only_regime{rid}.joblib")


# ---------------------------------------------------------------------------
# Training Orchestrator
# ---------------------------------------------------------------------------

def train_market(market: str, skip_timegpt: bool = False):
    """Train all 6 model variants for a single market."""
    print(f"\n{'='*70}")
    print(f"  PHASE 4: Training all models for {market}")
    print(f"{'='*70}")

    df = load_feature_matrix(market)
    train_df, val_df, test_df = temporal_split(df, market)

    os.makedirs(MODEL_DIR / market, exist_ok=True)
    os.makedirs(RESULTS_DIR / market, exist_ok=True)

    all_predictions = {}

    # ------------------------------------------------------------------
    # Model 4: XGBoost Only
    # ------------------------------------------------------------------
    print(f"\n--- [4/6] XGBoost Only ---")
    t0 = time.time()
    xgb_only = XGBoostOnlyModel(market=market)
    xgb_only.fit(train_df, val_df)
    xgb_only.save(str(MODEL_DIR / market / "xgb_only.joblib"))

    xgb_only_preds = xgb_only.predict(test_df)
    all_predictions["xgb_only"] = {
        "timestamps": test_df["hour"].values,
        "actual": test_df["LMP"].values,
        "predicted": xgb_only_preds,
        "regime_states": None,
    }
    print(f"    Done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Model 5: HMM + XGBoost (no LSTM)
    # ------------------------------------------------------------------
    print(f"\n--- [5/6] HMM + XGBoost ---")
    t0 = time.time()
    hmm_xgb = HMMXGBoostModel(market=market)
    hmm_xgb.fit(train_df, val_df)
    hmm_xgb.save(str(MODEL_DIR / market))

    hmm_xgb_preds, hmm_xgb_states = hmm_xgb.predict(test_df)
    test_aligned_len = len(hmm_xgb_preds)
    test_aligned = test_df.iloc[-test_aligned_len:]

    all_predictions["hmm_xgb"] = {
        "timestamps": test_aligned["hour"].values,
        "actual": test_aligned["LMP"].values,
        "predicted": hmm_xgb_preds,
        "regime_states": hmm_xgb_states,
    }
    print(f"    Done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Model 6: HMM + XGBoost + LSTM (full hybrid pipeline)
    # ------------------------------------------------------------------
    print(f"\n--- [6/6] HMM + XGBoost + LSTM (Full Hybrid) ---")
    t0 = time.time()
    hybrid = HybridPipeline(market=market, n_regimes=HMM_N_REGIMES)
    hybrid.train(train_df)
    hybrid.save(str(MODEL_DIR / market))

    hybrid_preds, hybrid_states = _predict_hybrid_on_test(hybrid, test_df, train_df)
    all_predictions["hybrid_full"] = {
        "timestamps": test_df["hour"].values[-len(hybrid_preds):],
        "actual": test_df["LMP"].values[-len(hybrid_preds):],
        "predicted": hybrid_preds,
        "regime_states": hybrid_states,
    }
    print(f"    Done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Models 2 & 3: TimeGPT
    # ------------------------------------------------------------------
    if not skip_timegpt:
        print(f"\n--- [2/6] TimeGPT Zero-Shot ---")
        print(f"--- [3/6] TimeGPT Fine-Tuned ---")
        _run_timegpt_benchmarks(market, train_df, test_df, all_predictions)
    else:
        print(f"\n--- Skipping TimeGPT (--skip-timegpt flag set) ---")
        for variant in ["timegpt_zero_shot", "timegpt_finetuned"]:
            path = RESULTS_DIR / market / f"{variant}_predictions.parquet"
            if path.exists():
                print(f"    Loading saved {variant} predictions from {path}")
                saved = pd.read_parquet(path)
                all_predictions[variant] = {
                    "timestamps": pd.to_datetime(saved["timestamp"]).values,
                    "actual": saved["actual"].values,
                    "predicted": saved["predicted"].values,
                    "regime_states": None,
                }

    # ------------------------------------------------------------------
    # Model 1: Naive Baseline
    # ------------------------------------------------------------------
    print(f"\n--- [1/6] Naive Baseline ---")
    all_predictions["naive_baseline"] = {
        "timestamps": test_df["hour"].values,
        "actual": test_df["LMP"].values,
        "predicted": None,
        "regime_states": None,
    }

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    print(f"\n--- Saving predictions ---")
    for model_name, data in all_predictions.items():
        if data["predicted"] is not None:
            pred_df = pd.DataFrame({
                "timestamp": data["timestamps"],
                "actual": data["actual"],
                "predicted": data["predicted"],
            })
            if data["regime_states"] is not None:
                states = data["regime_states"]
                if len(states) == len(pred_df):
                    pred_df["regime_state"] = states
            pred_df.to_parquet(
                RESULTS_DIR / market / f"{model_name}_predictions.parquet",
                index=False,
            )
            print(f"    Saved {model_name}: {len(pred_df):,} predictions")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print(f"\n--- Running evaluation ---")
    _evaluate_all(market, all_predictions)

    print(f"\n{'='*70}")
    print(f"  Phase 4 complete for {market}")
    print(f"  Artifacts saved to: {MODEL_DIR / market}")
    print(f"  Results saved to:   {RESULTS_DIR / market}")
    print(f"{'='*70}\n")


def _predict_hybrid_on_test(
    pipeline: HybridPipeline,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions from the full hybrid pipeline on the test set.
    Provides a lookback window for HMM and LSTM context.
    """
    lookback_hours = 168 + 24
    full_df = pd.concat([train_df.tail(lookback_hours), test_df], ignore_index=True)

    preds = []
    states = []

    for i in range(lookback_hours, len(full_df)):
        context = full_df.iloc[max(0, i - lookback_hours) : i + 1]
        try:
            result = pipeline.predict(context)
            preds.append(result["forecast_lmp"])
            states.append(result["current_regime"])
        except Exception as e:
            preds.append(context["LMP"].iloc[-2] if len(context) > 1 else 0.0)
            states.append(-1)

    return np.array(preds), np.array(states)


def _run_timegpt_benchmarks(
    market: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_predictions: dict,
):
    """Run TimeGPT zero-shot and fine-tuned benchmarks."""
    tgpt = TimeGPTBaseline()

    for variant, finetune_steps in [("timegpt_zero_shot", 0), ("timegpt_finetuned", 30)]:
        print(f"\n    Running {variant}...")
        window_preds = []
        window_actuals = []
        window_timestamps = []

        test_dates = pd.date_range(
            test_df["hour"].min().normalize(),
            test_df["hour"].max().normalize(),
            freq="24h",
        )

        for window_start in test_dates:
            window_end = window_start + pd.Timedelta(hours=24)
            context = train_df.tail(720)

            preceding_test = test_df[test_df["hour"] < window_start]
            if len(preceding_test) > 0:
                context = pd.concat([context, preceding_test]).tail(720)

            try:
                tgpt_context = context.copy()
                if "Time" not in tgpt_context.columns:
                    tgpt_context["Time"] = tgpt_context["hour"]

                if finetune_steps == 0:
                    fcst = tgpt.forecast_zero_shot(tgpt_context, horizon=24, market=market)
                else:
                    fcst = tgpt.forecast_finetuned(
                        tgpt_context, horizon=24, market=market,
                        finetune_steps=finetune_steps,
                    )

                actual_window = test_df[
                    (test_df["hour"] >= window_start) &
                    (test_df["hour"] < window_end)
                ]

                n = min(len(fcst), len(actual_window))
                if n > 0:
                    window_preds.extend(fcst["TimeGPT"].values[:n])
                    window_actuals.extend(actual_window["LMP"].values[:n])
                    window_timestamps.extend(actual_window["hour"].values[:n])

            except Exception as e:
                print(f"      Warning: {variant} failed for window {window_start}: {e}")
                continue

        if window_preds:
            all_predictions[variant] = {
                "timestamps": np.array(window_timestamps),
                "actual": np.array(window_actuals),
                "predicted": np.array(window_preds),
                "regime_states": None,
            }

            pred_df = pd.DataFrame({
                "timestamp": window_timestamps,
                "actual": window_actuals,
                "predicted": window_preds,
            })
            pred_df.to_parquet(
                RESULTS_DIR / market / f"{variant}_predictions.parquet",
                index=False,
            )
            print(f"      Saved {variant}: {len(pred_df):,} predictions")


def _evaluate_all(market: str, all_predictions: dict):
    """Compute metrics, run battery backtest, save comparison JSON."""
    comparator = ModelComparator()

    for model_name, data in all_predictions.items():
        if data["predicted"] is None:
            continue
        result = ModelResult(
            model_name=model_name,
            market=market,
            timestamps=pd.DatetimeIndex(data["timestamps"]),
            actual=data["actual"],
            predicted=data["predicted"],
            regime_states=data.get("regime_states"),
        )
        comparator.add_result(result)

    if "hybrid_full" in all_predictions and all_predictions["hybrid_full"]["regime_states"] is not None:
        regime_states = all_predictions["hybrid_full"]["regime_states"]
        for model_name in ["xgb_only", "timegpt_zero_shot", "timegpt_finetuned"]:
            if model_name in comparator.results:
                r = comparator.results[model_name]
                if len(regime_states) == len(r.actual):
                    r.regime_states = regime_states

    comparison = comparator.head_to_head()

    print(f"\n  {'='*60}")
    print(f"  LEADERBOARD — {market}")
    print(f"  {'='*60}")
    print(f"  {'Model':<25} {'RMSE':>8} {'MAE':>8} {'Dir.Acc':>8} {'N':>7}")
    print(f"  {'-'*60}")
    for m in comparison["leaderboard"]:
        print(
            f"  {m['model']:<25} "
            f"{m['rmse']:>8.2f} "
            f"{m['mae']:>8.2f} "
            f"{m['directional_accuracy']:>7.1%} "
            f"{m['n_samples']:>7,}"
        )

    print(f"\n  BATTERY ARBITRAGE P&L")
    print(f"  {'-'*60}")
    pnl_results = {}

    test_actuals = all_predictions["naive_baseline"]["actual"]
    naive_pnl = naive_baseline_pnl(test_actuals)
    pnl_results["naive_baseline"] = naive_pnl
    print(f"  {'naive_baseline':<25} P&L: ${naive_pnl['total_pnl']:>12,.2f}  Trades: {naive_pnl['num_trades']:>5}")

    for model_name, data in all_predictions.items():
        if data["predicted"] is None or model_name == "naive_baseline":
            continue
        pnl = backtest_strategy(
            actual_lmps=data["actual"],
            predicted_lmps=data["predicted"],
            model_name=model_name,
        )
        pnl_results[model_name] = pnl
        print(f"  {model_name:<25} P&L: ${pnl['total_pnl']:>12,.2f}  Trades: {pnl['num_trades']:>5}")

    comparison["pnl_results"] = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("cumulative_pnl", "hourly_pnl")}
        for k, v in pnl_results.items()
    }

    pnl_curves = {}
    for model_name, pnl in pnl_results.items():
        if "cumulative_pnl" in pnl:
            pnl_curves[model_name] = pnl["cumulative_pnl"]
    comparison["pnl_curves"] = pnl_curves

    output_path = RESULTS_DIR / market / "model_comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n  Comparison saved to: {output_path}")

    if comparison.get("regime_breakdown"):
        print(f"\n  REGIME-SEGMENTED BREAKDOWN")
        print(f"  {'-'*60}")
        for model_name, breakdown in comparison["regime_breakdown"].items():
            print(f"\n  {model_name}:")
            for regime in breakdown:
                print(
                    f"    Regime {regime['regime']}: "
                    f"RMSE={regime['rmse']:.2f}, "
                    f"MAE={regime['mae']:.2f}, "
                    f"N={regime['n_samples']:,} "
                    f"({regime['pct_of_total']:.1%})"
                )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Train & validate all models")
    parser.add_argument(
        "--market", type=str, default="both",
        choices=["CAISO", "ERCOT", "both"],
        help="Market to train (default: both)",
    )
    parser.add_argument(
        "--skip-timegpt", action="store_true",
        help="Skip TimeGPT API calls (use saved predictions if available)",
    )
    args = parser.parse_args()

    markets = ["CAISO", "ERCOT"] if args.market == "both" else [args.market]

    for market in markets:
        train_market(market, skip_timegpt=args.skip_timegpt)


if __name__ == "__main__":
    main()