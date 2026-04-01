import pandas as pd
import numpy as np
from dataclasses import dataclass

# Standardized result from any model
@dataclass
class ModelResult:
    model_name: str
    market: str
    timestamps: pd.DatetimeIndex
    actual: np.ndarray
    predicted: np.ndarray
    regime_states: np.ndarray = None

# Side-by-side evaluation of multiple forecasting models
# Outputs JSON convertible
class ModelComparator:

    def __init__(self):
        self.results = {}   # {model_name: ModelResult}
    
    def add_result(self, result: ModelResult):
        self.results[result.model_name] = result

    # Compute all metrics for a single model
    def compute_metrics(self, model_name: str) -> dict:
        r = self.results[model_name]
        errors = r.actual - r.predicted
        abs_errors = np.abs(errors)

        # Directional accuracy test
        actual_diff = np.diff(r.actual)
        pred_diff = np.diff(r.predicted)
        directional = np.mean(np.sign(actual_diff) == np.sign(pred_diff))

        return {
            "model": model_name,
            "market": r.market,
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(abs_errors)),
            "median_ae": float(np.median(abs_errors)),
            "max_error": float(np.max(abs_errors)),
            "directional_accuracy": float(directional),
            "n_samples": len(r.actual),
        }
    
    # Break down regime accuracy for custom models
    # Apply HMM regime to same time periods for TimeGPT for comparisons
    def compute_regime_breakdown(self, model_name: str) -> list[dict]:
        r = self.results[model_name]
        if r.regime_states is None:
            return []
        
        breakdown = []
        for regime_id in np.unique(r.regime_states):
            mask = r.regime_states == regime_id
            errors = r.actual[mask] - r.predicted[mask]
            breakdown.append({
                "regime": int(regime_id),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
                "n_samples": int(mask.sum()),
                "pct_of_total": float(mask.mean()),
            })
        return breakdown
    
    # Full comparison across all models
    # Returns structured data for the dashboard
    def head_to_head(self) -> dict:

        comparison = {
            "leaderboard": [],
            "regime_breakdown": {},
            "hourly_detail": [],
        }

        # Overall metrics leaderboard (sorted by RMSE)
        for name in self.results:
            metrics = self.compute_metrics(name)
            comparison["leaderboard"].append(metrics)
        comparison["leaderboard"].sort(key=lambda x: x["rmse"])

        # Regime breakdown for each model
        for name in self.results:
            breakdown = self.compute_regime_breakdown(name)
            if breakdown:
                comparison["regime_breakdown"][name] = breakdown

        # Hourly detail for the time series overlay chart
        # Build a timestamp-keyed dict from each model's predictions,
        # only keeping timestamps where ALL models have a prediction
        ts_sets = []
        ts_to_pred = {}
        for name, r in self.results.items():
            mapping = {ts: float(r.predicted[i]) for i, ts in enumerate(r.timestamps)}
            ts_to_pred[name] = mapping
            ts_sets.append(set(r.timestamps))

        # Intersect all timestamp sets so we compare apples-to-apples
        common_ts = sorted(set.intersection(*ts_sets))

        reference = list(self.results.values())[0]
        ref_actual = {ts: float(reference.actual[i]) for i, ts in enumerate(reference.timestamps)}

        for ts in common_ts:
            point = {"timestamp": ts.isoformat(), "actual": ref_actual[ts]}
            for name in self.results:
                point[f"pred_{name}"] = ts_to_pred[name][ts]
            comparison["hourly_detail"].append(point)

        return comparison
    
    # Apply the HMM's regime labels to TimeGPT's test period
    # This lets us compare accuracy per regime even for models that don't have regime detection
    def apply_regime_labels_to_timegpt(
            timegpt_result: ModelResult,
            regime_detector,
            feature_df: pd.DataFrame,
    ) -> ModelResult:
        X_hmm = regime_detector.prepare_observations(feature_df)
        states = regime_detector.model.predict(X_hmm)
        # Align with test period
        aligned_states = states[-len(timegpt_result.actual):]
        timegpt_result.regime_states = aligned_states
        return timegpt_result