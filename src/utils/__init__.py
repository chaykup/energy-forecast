"""
Shared utilities and configuration.

    config — Environment variables, market configs, model hyperparameters, paths
"""

from src.utils.config import (
    EIA_API_KEY,
    FRED_API_KEY,
    NIXTLA_API_KEY,
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    MARKETS,
    MARKET_CONFIG,
    SPLIT_CONFIG,
    HMM_N_REGIMES,
    XGB_PARAMS,
    LSTM_PARAMS,
    EXCLUDE_COLS,
    BATTERY_CONFIG,
)