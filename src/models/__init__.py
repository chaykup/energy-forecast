"""
Forecasting models.

    RegimeDetector   — HMM (hmmlearn) for market regime classification
    RegimeXGBoost    — Per-regime XGBoost tabular model
    RegimeLSTM       — Per-regime LSTM residual corrector (PyTorch)
    HybridPipeline   — Full orchestration: regime → XGBoost → LSTM → forecast
    TimeGPTBaseline  — Nixtla TimeGPT zero-shot + fine-tuned benchmark
"""

from src.models.hmm_regime import RegimeDetector
from src.models.xgboost_model import RegimeXGBoost
from src.models.lstm_model import RegimeLSTM
from src.models.hybrid_pipeline import HybridPipeline
from src.models.timegpt_baseline import TimeGPTBaseline

__all__ = [
    "RegimeDetector",
    "RegimeXGBoost",
    "RegimeLSTM",
    "HybridPipeline",
    "TimeGPTBaseline",
]