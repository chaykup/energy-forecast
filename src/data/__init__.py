"""
Data ingestion and feature engineering.

Clients:
    LMPClient        — CAISO + ERCOT LMPs via gridstatus
    EIAClient        — Hourly demand + generation mix via EIA API v2
    WeatherClient    — Temperature, solar irradiance, wind via Open-Meteo
    FREDClient       — Henry Hub gas price, macro indicators via FRED API

Feature Engineering:
    FeatureEngineer  — Merges all sources into hourly feature matrix
"""

from src.data.gridstatus_client import LMPClient
from src.data.eia_client import EIAClient
from src.data.weather_client import WeatherClient
from src.data.fred_client import FREDClient

__all__ = [
    "LMPClient",
    "EIAClient",
    "WeatherClient",
    "FREDClient",
    "FeatureEngineer",
]