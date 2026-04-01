import requests
import pandas as pd

# Pulls weather data and forecasts from Open-Meteo API
class WeatherClient:

    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Coordinates for each market
    LOCATIONS = {
        "CAISO": {"latitude": 34.05, "longitude": -118.24, "name": "Los Angeles"},
        "ERCOT": {"latitude": 29.76, "longitude": -95.37, "name": "Houston"},
    }

    HOURLY_VARS = {
        "temperature_2m",
        "relative_humidity_2m",
        "direct_radiation",     # Solar irradiance
        "wind_speed_10m",
        "cloud_cover",
    }

    # Get hourly weather forecast for next N days
    def get_forecast(self, market: str, forecast_days: int = 2) -> pd.DataFrame:
        loc = self.LOCATIONS[market]
        params = {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "hourly": ",".join(self.HOURLY_VARS),
            "forecast_days": forecast_days,
            "timezone": "America/Los_Angeles" if market == "CAISO" else "America/Chicago",
        }
        resp = requests.get(self.BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df["market"] = market
        return df
    
    # Get historical hourly weather data for model training
    def get_historical(self, market: str, start_date: str, end_date: str) -> pd.DataFrame:
        loc = self.LOCATIONS[market]
        params = {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "hourly": ",".join(self.HOURLY_VARS),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "America/Los_Angeles" if market == "CAISO" else "America/Chicago",
        }
        resp = requests.get(self.HISTORICAL_URL, params=params)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df["market"] = market
        return df