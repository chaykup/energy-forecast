"""
Model evaluation and comparison.

    metrics          — RMSE, MAE, MAPE, directional accuracy, spike capture rate
    ModelComparator  — Side-by-side evaluation across all model variants
    ModelResult      — Standardized prediction container for any model
    backtest_strategy    — Battery arbitrage P&L simulation (model-driven)
    naive_baseline_pnl   — Naive heuristic: charge overnight, discharge peak
"""

from src.evaluation.metrics import compute_all_metrics, compute_regime_metrics
from src.evaluation.model_comparator import ModelComparator, ModelResult
from src.evaluation.backtest import backtest_strategy, naive_baseline_pnl

__all__ = [
    "compute_all_metrics",
    "compute_regime_metrics",
    "ModelComparator",
    "ModelResult",
    "backtest_strategy",
    "naive_baseline_pnl",
]