# src/training/walk_forward.py
"""
Walk-Forward Validation for Time Series Models

Implements expanding-window walk-forward cross-validation as specified
in Phase 4.1 of the dev guide. This is the gold standard for evaluating
time series forecasting models because it:

  1. Never leaks future data into training
  2. Tests on progressively later periods (captures regime drift)
  3. Uses expanding training windows (mimics production retraining)

Fold structure (monthly steps, configurable):
    Fold 1: Train Jan-Jun 2023  → Validate Jul 2023  (24h windows)
    Fold 2: Train Jan-Jul 2023  → Validate Aug 2023
    Fold 3: Train Jan-Aug 2023  → Validate Sep 2023
    ...expanding window until validation period is exhausted
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.evaluation.model_comparator import ModelComparator, ModelResult


@dataclass
class WalkForwardFold:
    """Single fold result from walk-forward validation."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_train: int
    n_val: int
    actual: np.ndarray
    predicted: np.ndarray
    regime_states: Optional[np.ndarray] = None
    rmse: float = 0.0
    mae: float = 0.0
    directional_accuracy: float = 0.0
    train_time_seconds: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    model_name: str
    market: str
    folds: list[WalkForwardFold] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([f.rmse for f in self.folds]))

    @property
    def std_rmse(self) -> float:
        return float(np.std([f.rmse for f in self.folds]))

    @property
    def mean_mae(self) -> float:
        return float(np.mean([f.mae for f in self.folds]))

    @property
    def mean_directional_accuracy(self) -> float:
        return float(np.mean([f.directional_accuracy for f in self.folds]))

    @property
    def all_actual(self) -> np.ndarray:
        """Concatenated actuals across all folds."""
        return np.concatenate([f.actual for f in self.folds])

    @property
    def all_predicted(self) -> np.ndarray:
        """Concatenated predictions across all folds."""
        return np.concatenate([f.predicted for f in self.folds])

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "market": self.market,
            "n_folds": self.n_folds,
            "mean_rmse": self.mean_rmse,
            "std_rmse": self.std_rmse,
            "mean_mae": self.mean_mae,
            "mean_directional_accuracy": self.mean_directional_accuracy,
            "per_fold_rmse": [f.rmse for f in self.folds],
            "per_fold_mae": [f.mae for f in self.folds],
            "per_fold_n_val": [f.n_val for f in self.folds],
            "per_fold_train_time_s": [f.train_time_seconds for f in self.folds],
        }

    def print_summary(self):
        print(f"\n  Walk-Forward Validation: {self.model_name} ({self.market})")
        print(f"  {'─'*55}")
        print(f"  Folds: {self.n_folds}")
        print(f"  RMSE:  {self.mean_rmse:.2f} ± {self.std_rmse:.2f}")
        print(f"  MAE:   {self.mean_mae:.2f}")
        print(f"  Dir. Accuracy: {self.mean_directional_accuracy:.1%}")
        print(f"  {'─'*55}")
        print(f"  {'Fold':>6} {'Train Period':>25} {'Val Period':>25} {'RMSE':>8} {'MAE':>8} {'N':>6}")
        for f in self.folds:
            print(
                f"  {f.fold_id:>6} "
                f"{str(f.train_start.date()):>12}→{str(f.train_end.date()):>12} "
                f"{str(f.val_start.date()):>12}→{str(f.val_end.date()):>12} "
                f"{f.rmse:>8.2f} "
                f"{f.mae:>8.2f} "
                f"{f.n_val:>6}"
            )


class WalkForwardValidator:
    """
    Walk-forward cross-validation engine.

    Usage:
        validator = WalkForwardValidator(
            min_train_months=6,
            val_window_months=1,
            step_months=1,
        )
        result = validator.validate(
            df=feature_matrix,
            market="CAISO",
            model_name="hybrid_full",
            train_fn=my_train_function,
            predict_fn=my_predict_function,
        )
        result.print_summary()
    """

    def __init__(
        self,
        min_train_months: int = 6,
        val_window_months: int = 1,
        step_months: int = 1,
        max_folds: int = 20,
    ):
        self.min_train_months = min_train_months
        self.val_window_months = val_window_months
        self.step_months = step_months
        self.max_folds = max_folds

    def generate_folds(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generate (train_start, train_end, val_start, val_end) tuples.
        Uses expanding window: training always starts from the beginning.
        """
        data_start = df["Time"].min()
        data_end = df["Time"].max()

        # First validation window starts after min_train_months
        first_val_start = data_start + pd.DateOffset(months=self.min_train_months)

        folds = []
        val_start = first_val_start

        while val_start < data_end and len(folds) < self.max_folds:
            train_start = data_start
            train_end = val_start - pd.Timedelta(hours=1)  # No overlap
            val_end = min(
                val_start + pd.DateOffset(months=self.val_window_months) - pd.Timedelta(hours=1),
                data_end,
            )

            if val_end <= val_start:
                break

            folds.append((train_start, train_end, val_start, val_end))
            val_start += pd.DateOffset(months=self.step_months)

        return folds

    def validate(
        self,
        df: pd.DataFrame,
        market: str,
        model_name: str,
        train_fn: Callable,
        predict_fn: Callable,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            df: Full feature matrix with 'Time' and 'LMP' columns.
            market: "CAISO" or "ERCOT"
            model_name: Name for results tracking.
            train_fn: Callable(train_df) -> trained_model
                Takes training data, returns a fitted model object.
            predict_fn: Callable(model, val_df) -> (predictions, regime_states_or_None)
                Takes model + validation data, returns predictions array
                and optionally regime states.

        Returns:
            WalkForwardResult with per-fold and aggregate metrics.
        """
        folds_spec = self.generate_folds(df)
        result = WalkForwardResult(model_name=model_name, market=market)

        print(f"\n  Walk-forward validation: {model_name} ({market})")
        print(f"  {len(folds_spec)} folds, expanding window")

        import time

        for i, (train_start, train_end, val_start, val_end) in enumerate(folds_spec):
            train_df = df[(df["Time"] >= train_start) & (df["Time"] <= train_end)].copy()
            val_df = df[(df["Time"] >= val_start) & (df["Time"] <= val_end)].copy()

            if len(train_df) < 100 or len(val_df) < 24:
                print(f"    Fold {i+1}: Skipping (insufficient data)")
                continue

            t0 = time.time()

            # Train
            model = train_fn(train_df)

            # Predict
            preds_output = predict_fn(model, val_df)
            if isinstance(preds_output, tuple):
                predictions, regime_states = preds_output
            else:
                predictions, regime_states = preds_output, None

            train_time = time.time() - t0

            # Align lengths (HMM may drop initial rows)
            actual = val_df["LMP"].values
            n = min(len(actual), len(predictions))
            actual = actual[-n:]
            predictions = predictions[-n:]
            if regime_states is not None:
                regime_states = regime_states[-n:]

            # Compute fold metrics
            errors = actual - predictions
            rmse = float(np.sqrt(np.mean(errors**2)))
            mae = float(np.mean(np.abs(errors)))

            actual_diff = np.diff(actual)
            pred_diff = np.diff(predictions)
            dir_acc = float(np.mean(np.sign(actual_diff) == np.sign(pred_diff))) if len(actual_diff) > 0 else 0.0

            fold = WalkForwardFold(
                fold_id=i + 1,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                n_train=len(train_df),
                n_val=n,
                actual=actual,
                predicted=predictions,
                regime_states=regime_states,
                rmse=rmse,
                mae=mae,
                directional_accuracy=dir_acc,
                train_time_seconds=train_time,
            )
            result.folds.append(fold)

            print(
                f"    Fold {i+1}/{len(folds_spec)}: "
                f"RMSE={rmse:.2f}, MAE={mae:.2f}, "
                f"Dir.Acc={dir_acc:.1%}, "
                f"N_train={len(train_df):,}, N_val={n:,}, "
                f"Time={train_time:.1f}s"
            )

        return result


def run_walk_forward_all_models(
    df: pd.DataFrame,
    market: str,
    min_train_months: int = 6,
) -> dict[str, WalkForwardResult]:
    """
    Convenience function: run walk-forward validation for all custom model variants.

    Returns dict mapping model_name → WalkForwardResult.
    """
    from src.models.hmm_regime import RegimeDetector
    from src.models.xgboost_model import RegimeXGBoost
    from src.models.lstm_model import RegimeLSTM
    from src.models.hybrid_pipeline import HybridPipeline

    validator = WalkForwardValidator(min_train_months=min_train_months)
    results = {}

    # Columns to exclude from features
    exclude = [
        "LMP", "Time", "hour", "iso", "Location", "Market",
        "Location Name", "Location Type", "regime",
    ]

    def get_features(d):
        return [c for c in d.columns if c not in exclude]

    # --- XGBoost Only ---
    def train_xgb_only(train_df):
        import xgboost as xgb
        feature_cols = get_features(train_df)
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", tree_method="hist", random_state=42,
        )
        model.fit(train_df[feature_cols], train_df["LMP"], verbose=False)
        return (model, feature_cols)

    def predict_xgb_only(model_tuple, val_df):
        model, feature_cols = model_tuple
        return model.predict(val_df[feature_cols])

    results["xgb_only"] = validator.validate(
        df, market, "xgb_only", train_xgb_only, predict_xgb_only
    )

    # --- HMM + XGBoost ---
    def train_hmm_xgb(train_df):
        from src.training.train_all_models import HMMXGBoostModel
        model = HMMXGBoostModel(market=market)
        # Use last 20% of training data as pseudo-validation
        split_idx = int(len(train_df) * 0.8)
        pseudo_val = train_df.iloc[split_idx:]
        model.fit(train_df.iloc[:split_idx], pseudo_val)
        return model

    def predict_hmm_xgb(model, val_df):
        return model.predict(val_df)  # Returns (preds, states)

    results["hmm_xgb"] = validator.validate(
        df, market, "hmm_xgb", train_hmm_xgb, predict_hmm_xgb
    )

    # --- Full Hybrid (HMM + XGBoost + LSTM) ---
    def train_hybrid(train_df):
        pipeline = HybridPipeline(market=market, n_regimes=3)
        pipeline.train(train_df)
        return pipeline

    def predict_hybrid(pipeline, val_df):
        X_hmm = pipeline.regime_detector.prepare_observations(val_df)
        states = pipeline.regime_detector.model.predict(X_hmm)
        val_aligned = val_df.iloc[-len(states):].copy()

        preds = np.zeros(len(val_aligned))
        for rid in range(pipeline.n_regimes):
            mask = states == rid
            if mask.sum() == 0:
                continue
            regime_df = val_aligned[mask]
            xgb_pred = pipeline.xgb_models[rid].predict(regime_df)

            # LSTM residual correction
            residuals = pipeline.xgb_models[rid].get_residuals(val_aligned, states)
            for j, idx in enumerate(np.where(mask)[0]):
                if len(residuals) >= pipeline.lstm_models[rid].seq_len:
                    lstm_correction = pipeline.lstm_models[rid].predict(residuals.values[:idx+1])
                else:
                    lstm_correction = 0.0
                preds[idx] = xgb_pred[j] + lstm_correction

        return preds, states

    results["hybrid_full"] = validator.validate(
        df, market, "hybrid_full", train_hybrid, predict_hybrid
    )

    # Print comparative summary
    print(f"\n  {'='*60}")
    print(f"  WALK-FORWARD COMPARISON — {market}")
    print(f"  {'='*60}")
    print(f"  {'Model':<20} {'RMSE':>10} {'±σ':>8} {'MAE':>8} {'Dir.Acc':>8}")
    print(f"  {'─'*60}")
    for name, res in results.items():
        print(
            f"  {name:<20} "
            f"{res.mean_rmse:>10.2f} "
            f"{res.std_rmse:>8.2f} "
            f"{res.mean_mae:>8.2f} "
            f"{res.mean_directional_accuracy:>7.1%}"
        )

    return results