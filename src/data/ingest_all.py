import argparse
import pandas as pd
from pathlib import Path
from src.data.gridstatus_client import LMPClient
from src.data.eia_client import EIAClient
from src.data.weather_client import WeatherClient
from src.data.fred_client import FREDClient

RAW_DIR = Path("data/raw")
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
ERCOT_START = "2023-05-01"


def fetch_lmp_chunked(lmp_client: LMPClient, market: str,
                      start: str, end: str) -> pd.DataFrame:
    """Fetch LMPs in monthly chunks to avoid OASIS API limits."""
    all_dfs = []
    date_range = pd.date_range(start=start, end=end, freq="MS")

    for i, month_start in enumerate(date_range):
        if i + 1 < len(date_range):
            month_end = date_range[i + 1]
        else:
            month_end = pd.Timestamp(end)

        start_str = month_start.strftime("%b %d, %Y")
        end_str = month_end.strftime("%b %d, %Y")
        print(f"    {start_str} → {end_str}...", end=" ")

        try:
            if market == "CAISO":
                chunk = lmp_client.get_caiso_lmp(start=start_str, end=end_str)
            else:
                chunk = lmp_client.get_ercot_lmp(start=start_str, end=end_str)

            if len(chunk) > 0:
                all_dfs.append(chunk)
                print(f"{len(chunk)} rows")
            else:
                print("0 rows (skipping)")
        except Exception as e:
            print(f"error: {e} (skipping)")

    if not all_dfs:
        raise ValueError(f"No LMP data retrieved for {market} {start}–{end}")
    return pd.concat(all_dfs).reset_index(drop=True)


def ingest_market(market: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    m = market.upper()
    start = ERCOT_START if m == "ERCOT" else START_DATE

    print(f"\n{'=' * 60}")
    print(f"    Ingesting {m}: {start} to {END_DATE}")
    print(f"{'=' * 60}")

    # LMP data (chunked monthly)
    print(f"\n[1/4] Fetching {m} LMPs (monthly chunks)...")
    lmp_client = LMPClient()
    lmps = fetch_lmp_chunked(lmp_client, m, start, END_DATE)
    lmps.to_parquet(RAW_DIR / f"{m.lower()}_lmps.parquet")
    print(f"    Total: {len(lmps)} rows saved")

    # EIA demand
    print(f"\n[2/4] Fetching {m} hourly demand...")
    eia = EIAClient()
    demand = eia.get_hourly_demand(m, start, END_DATE)
    demand.to_parquet(RAW_DIR / f"{m.lower()}_demand.parquet")
    print(f"    Saved {len(demand)} rows")

    # Generation mix
    print(f"\n[3/4] Fetching {m} generation mix...")
    gen_mix = eia.get_hourly_generation_by_fuel(m, start, END_DATE)
    gen_mix.to_parquet(RAW_DIR / f"{m.lower()}_gen_mix.parquet")
    print(f"    Saved {len(gen_mix)} rows")

    # Weather
    print(f"\n[4/4] Fetching {m} historical weather...")
    weather = WeatherClient()
    wx = weather.get_historical(m, start, END_DATE)
    wx.to_parquet(RAW_DIR / f"{m.lower()}_weather.parquet")
    print(f"    Saved {len(wx)} rows")

    print(f"\n✓ {m} ingestion complete")


def ingest_macro():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "macro_indicators.parquet"
    if path.exists():
        print("\n[Macro] Already exists, skipping (delete to re-fetch)")
        return
    print("\n[Macro] Fetching FRED macro indicators...")
    fred = FREDClient()
    macro = fred.get_all_macro(START_DATE, END_DATE)
    macro.to_parquet(path)
    print(f"    Saved {len(macro)} rows")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Ingest raw data from APIs")
    parser.add_argument("--market", required=True, choices=["CAISO", "ERCOT", "both"])
    args = parser.parse_args()
    ingest_macro()
    if args.market == "both":
        ingest_market("CAISO")
        ingest_market("ERCOT")
    else:
        ingest_market(args.market)

if __name__ == "__main__":
    main()