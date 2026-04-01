
"""
Loads API keys from .env, defines market-specific constants,
model hyperparameters, file paths, and shared column exclusions.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

load_dotenv()

EIA_API_KEY = os.getenv("EIA_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
NIXTLA_API_KEY = os.getenv("NIXTLA_API_KEY")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # energy-forecast/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

# ---------------------------------------------------------------------------
# Market configurations
# ---------------------------------------------------------------------------

MARKETS = ["CAISO", "ERCOT"]

MARKET_CONFIG = {
    "CAISO": {
        "eia_respondent": "CISO",
        "timezone": "America/Los_Angeles",
        "weather_lat": 34.05,
        "weather_lon": -118.24,
        "weather_city": "Los Angeles",
        "caiso_locations": [
            "TH_NP15_GEN-APND",   # Northern California
            "TH_SP15_GEN-APND",   # Southern California
            "TH_ZP26_GEN-APND",   # Zone P26 (transmission hub)
        ],
        "regime_labels": {0: "solar_surplus", 1: "normal", 2: "peak_stress"},
    },
    "ERCOT": {
        "eia_respondent": "ERCO",
        "timezone": "America/Chicago",
        "weather_lat": 29.76,
        "weather_lon": -95.37,
        "weather_city": "Houston",
        "caiso_locations": None,  # ERCOT uses settlement point hubs, not CAISO nodes
        "regime_labels": {0: "normal", 1: "high_demand", 2: "scarcity"},
    },
}

# ---------------------------------------------------------------------------
# FRED series IDs
# ---------------------------------------------------------------------------

FRED_SERIES = {
    "DHHNGSP": "Henry Hub Natural Gas Spot Price (daily)",
    "DCOILWTICO": "WTI Crude Oil Price (daily)",
    "DFF": "Federal Funds Effective Rate (daily)",
    "DTWEXBGS": "Trade Weighted US Dollar Index (daily)",
}

# ---------------------------------------------------------------------------
# EIA fuel type codes
# ---------------------------------------------------------------------------

EIA_FUEL_CODES = {
    "SUN": "Solar",
    "WND": "Wind",
    "NG": "Natural Gas",
    "NUC": "Nuclear",
    "WAT": "Hydro",
    "COL": "Coal",
    "OIL": "Oil",
    "OTH": "Other",
}

# ---------------------------------------------------------------------------
# Temporal split boundaries
# ---------------------------------------------------------------------------

SPLIT_CONFIG = {
    "CAISO": {
        "train_end": "2024-03-31",
        "val_end": "2024-07-31",
        # Test: 2024-08-01 → end of dataset
    },
    "ERCOT": {
        "train_end": "2024-03-31",
        "val_end": "2024-07-31",
    },
}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

HMM_N_REGIMES = 3

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

LSTM_PARAMS = {
    "seq_len": 24,
    "hidden_size": 64,
    "num_layers": 2,
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 32,
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Columns excluded from model training (metadata, target, etc.)
EXCLUDE_COLS = [
    "LMP", "Time", "hour", "iso", "Location", "Market",
    "Location Name", "Location Type", "regime",
]

# Lag horizons for LMP features (in hours)
LMP_LAG_HOURS = [1, 2, 3, 4, 6, 12, 24, 48, 168]  # 168 = same hour last week

# Rolling window sizes for LMP statistics (in hours)
LMP_ROLLING_WINDOWS = [24, 168]

# ---------------------------------------------------------------------------
# Battery arbitrage parameters
# ---------------------------------------------------------------------------

BATTERY_CONFIG = {
    "capacity_mwh": 100,
    "max_charge_rate_mw": 25,
    "round_trip_efficiency": 0.90,
    "min_price_spread": 5.0,  # Minimum $/MWh spread to act
}

# ---------------------------------------------------------------------------
# Open-Meteo weather variables
# ---------------------------------------------------------------------------

WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "direct_radiation",      # Solar irradiance (W/m²)
    "wind_speed_10m",
    "cloud_cover",
]