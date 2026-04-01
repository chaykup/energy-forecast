import requests
import pandas as pd
from src.utils.config import EIA_API_KEY

# Pulls hourly electricity demand and generation data from Energy Information Administration API v2
class EIAClient:

    BASE_URL = "https://api.eia.gov/v2"

    RESPONDENT = {"CAISO": "CISO", "ERCOT": "ERCO"}

    def __init__(self):
        self.api_key = EIA_API_KEY

    # respondant: "CISO" (CAISO) OR "ERCO" (ERCOT)
    # Route: electricity/rto/region-data/data
    def get_hourly_demand(self, market: str, start: str, end: str) -> pd.DataFrame:
        url = f"{self.BASE_URL}/electricity/rto/region-data/data/"
        params = {
            "api_key": self.api_key, 
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": self.RESPONDENT[market],
            "facets[type][]": "D", # D = demand
            "start": f"{start}T00",
            "end": f"{end}T00",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }
        return self._paginated_fetch(url, params)
    
    # respondant: "CISO" (CAISO) OR "ERCO" (ERCOT)
    # Route: electricity/rto/fuel-type-data/data
    # Fuel types: SUN, WND, NG, NUC, WAT, COL, OIL, OTH
    def get_hourly_generation_by_fuel(self, market: str, start: str, end: str) -> pd.DataFrame:
        url = f"{self.BASE_URL}/electricity/rto/fuel-type-data/data/"
        params = {
            "api_key": self.api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": self.RESPONDENT[market],
            "start": f"{start}T00",
            "end": f"{end}T00",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }
        return self._paginated_fetch(url, params)
    
    def _paginated_fetch(self, url: str, params: dict) -> pd.DataFrame:
        all_data = []
        offset = 0
        while True:
            params["offset"] = offset
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("response", {}).get("data", [])
            if not records:
                break
            all_data.extend(records)
            if len(records) < 5000:
                break
            offset += 5000
        return pd.DataFrame(all_data)