import os
import gridstatus
from gridstatus.ercot_api.ercot_api import ErcotAPI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class LMPClient:
    # Key pricing nodes
    CAISO_LOCATIONS = [
        "TH_NP15_GEN-APND",  # Northern CA
        "TH_SP15_GEN-APND",  # Southern CA
        "TH_ZP26_GEN-APND",  # Zone P26 (transmission hub)
    ]

    # ERCOT hub settlement points (analogous to CAISO trading hubs)
    ERCOT_LOCATIONS = [
        "HB_HOUSTON",
        "HB_NORTH",
        "HB_SOUTH",
        "HB_WEST",
    ]

    def __init__(self):
        self.caiso = gridstatus.CAISO()
        # ErcotAPI reads credentials from env vars automatically:
        #   ERCOT_API_USERNAME, ERCOT_API_PASSWORD,
        #   ERCOT_PUBLIC_API_SUBSCRIPTION_KEY
        self.ercot_api = ErcotAPI()

    # Fetch CAISO LMPs
    # Markets: DAY_AHEAD_HOURLY, REAL_TIME_5_MIN, REAL_TIME_15_MIN
    # Returns: DataFrame with columns [Time, Market, Location, Location Name,
    #          Location Type, LMP, Energy, Congestion, Loss]
    def get_caiso_lmp(
        self, start: str, end: str, market: str = "DAY_AHEAD_HOURLY"
    ) -> pd.DataFrame:
        df = self.caiso.get_lmp(
            start=start,
            end=end,
            market=market,
            locations=self.CAISO_LOCATIONS,
        )
        df["iso"] = "CAISO"
        return df

    # Fetch ERCOT DAM hourly LMPs via ErcotAPI.
    # Uses get_spp_day_ahead_hourly() which hits /np4-190-cd/dam_stlmnt_pnt_prices
    # This returns hourly DAM settlement point prices — the correct ERCOT
    # equivalent to CAISO DAY_AHEAD_HOURLY for a fair cross-market benchmark.
    # Requires ERCOT API credentials in .env.
    #
    # Output columns from get_spp_day_ahead_hourly():
    #   [Time, Interval Start, Interval End, Location, Location Type, Market, SPP]
    # "Time" already exists — only need to rename SPP → LMP for feature_engineering.py
    def get_ercot_lmp(self, start: str, end: str) -> pd.DataFrame:
        df = self.ercot_api.get_spp_day_ahead_hourly(
            date=start,
            end=end,
        )

        # Filter to hubs only (mirrors CAISO's 3 trading hubs)
        df = df[df["Location"].isin(self.ERCOT_LOCATIONS)]

        # Rename SPP → LMP to match CAISO schema for feature_engineering.py
        # "Time" column already exists from the parser, no rename needed
        df = df.rename(columns={"SPP": "LMP"})

        df["iso"] = "ERCOT"
        return df

    def get_latest(self, iso: str) -> pd.DataFrame:
        if iso == "CAISO":
            return self.caiso.get_lmp(
                date="today",
                market="DAY_AHEAD_HOURLY",
                locations=self.CAISO_LOCATIONS,
            )
        elif iso == "ERCOT":
            df = self.ercot_api.get_spp_day_ahead_hourly(date="today")
            df = df[df["Location"].isin(self.ERCOT_LOCATIONS)]
            df = df.rename(columns={"SPP": "LMP"})
            return df