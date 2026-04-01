# src/evaluation/metrics.py
"""
Evaluation metrics for LMP forecasting models.

All metrics from Phase 4.2 of the dev guide:
  - RMSE: Overall forecast accuracy in $/MWh
  - MAE: Average absolute error (less sensitive to spikes)
  - Median AE: Robust central tendency of errors
  - Max Error: Worst-case miss (critical for spike regimes)
  - MAPE: Mean absolute percentage error (scale-independent)
  - Directional Accuracy: % of hours where predicted direction was correct
"""

import numpy as np
import pandas as pd
from typing import Optional


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error in $/MWh."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error in $/MWh."""
    return float(np.mean(np.abs(actual - predicted)))


def median_ae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Median Absolute Error — robust to outliers."""
    return float(np.median(np.abs(actual - predicted)))


def max_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Maximum absolute error — worst-case miss."""
    return float(np.max(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error.

    eps prevents division by zero for near-zero LMPs (common in CAISO
    solar surplus regimes where prices can go negative).
    """
    return float(np.mean(np.abs(actual - predicted) / (np.abs(actual) + eps)) * 100)


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Percentage of hours where the predicted direction of price movement
    matched the actual direction.

    This matters for battery arbitrage: even if the magnitude is wrong,
    getting the direction right means charge/discharge timing is correct.
    """
    actual_diff = np.diff(actual)
    pred_diff = np.diff(predicted)
    if len(actual_diff) == 0:
        return 0.0
    return float(np.mean(np.sign(actual_diff) == np.sign(pred_diff)))


def spike_capture_rate(
    actual: np.ndarray,
    predicted: np.ndarray,
    threshold_percentile: float = 95,
) -> float:
    """
    What fraction of actual price spikes did the model anticipate?

    A "spike" is any hour where actual LMP exceeds the given percentile.
    The model "captured" it if the prediction was also above the median
    (i.e., the model knew prices would be elevated).
    """
    threshold = np.percentile(actual, threshold_percentile)
    median_pred = np.median(predicted)
    spike_mask = actual >= threshold
    if spike_mask.sum() == 0:
        return 0.0
    captured = predicted[spike_mask] >= median_pred
    return float(captured.mean())


def negative_price_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Accuracy on hours where actual prices were negative.
    CAISO-specific: solar surplus drives negative LMPs.
    """
    neg_mask = actual < 0
    if neg_mask.sum() == 0:
        return float("nan")
    # Did the model predict negative or near-zero?
    correctly_low = predicted[neg_mask] < 10  # Within $10 of zero
    return float(correctly_low.mean())


def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "",
    market: str = "",
) -> dict:
    """
    Compute the full metrics suite from Phase 4.2.
    Returns a flat dict suitable for JSON serialization or DataFrame row.
    """
    return {
        "model": model_name,
        "market": market,
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "median_ae": median_ae(actual, predicted),
        "max_error": max_error(actual, predicted),
        "mape": mape(actual, predicted),
        "directional_accuracy": directional_accuracy(actual, predicted),
        "spike_capture_rate_95": spike_capture_rate(actual, predicted, 95),
        "negative_price_accuracy": negative_price_accuracy(actual, predicted),
        "n_samples": len(actual),
        "mean_actual": float(np.mean(actual)),
        "std_actual": float(np.std(actual)),
        "mean_predicted": float(np.mean(predicted)),
        "std_predicted": float(np.std(predicted)),
    }


def compute_regime_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    regime_states: np.ndarray,
    regime_labels: Optional[dict] = None,
    model_name: str = "",
) -> list[dict]:
    """
    Break down metrics by regime state.

    This is the "most interesting metric for a technical interviewer":
    if the custom model beats TimeGPT during spike regimes where the
    most money is made/lost, that's the key insight.
    """
    results = []
    for rid in np.unique(regime_states):
        mask = regime_states == rid
        if mask.sum() < 2:
            continue
        label = regime_labels.get(int(rid), f"regime_{rid}") if regime_labels else f"regime_{rid}"
        metrics = compute_all_metrics(actual[mask], predicted[mask])
        metrics["regime_id"] = int(rid)
        metrics["regime_label"] = label
        metrics["model"] = model_name
        metrics["pct_of_total"] = float(mask.mean())
        results.append(metrics)
    return results


def metrics_to_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """Convert list of metrics dicts into a clean comparison DataFrame."""
    return pd.DataFrame(metrics_list).set_index("model")