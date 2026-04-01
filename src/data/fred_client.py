import requests
import pandas as pd
from src.utils.config import FRED_API_KEY

# Pulls macroeconomic indicators from FRED API
class FREDClient:

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Key features for regime detection
    SERIES = {
        "DHHNGSP": "Henry Hub Natural Gas Spot Price (daily)",
        "DCOILWTICO": "WTI Crude Oil Price (daily)",
        "DFF": "Federal Funds Effective Rate (daily)",
        "DTWEXBGS": "Trade Weighted US Dollar Index (daily)",
    }

    def __init__(self):
        self.api_key = FRED_API_KEY

    # Fetch a single FRED series
    # Start / End format: YYYY-MM-DD
    def get_series(self, series_id: str, start: str, end: str) -> pd.DataFrame:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
        resp = requests.get(self.BASE_URL,params=params)
        resp.raise_for_status()
        obs = resp.json()["observations"]
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value"]].rename(columns={"value": series_id})
        return df
    
    # Fetch all macro series and merge on date
    def get_all_macro(self, start: str, end: str) -> pd.DataFrame:
        dfs = [self.get_series(sid, start, end) for sid in self.SERIES]
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="date", how="outer")
        merged = merged.sort_values("date").ffill() # forward-fill missing days
        return merged