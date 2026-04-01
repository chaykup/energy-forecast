import argparse
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def to_naive_utc(dt) -> pd.DatetimeIndex:
    """Normalize any datetime Series or Index to timezone-naive UTC."""
    if isinstance(dt, pd.Series):
        if dt.dt.tz is not None:
            return dt.dt.tz_convert("UTC").dt.tz_localize(None)
        return dt
    else:  # DatetimeIndex
        if dt.tz is not None:
            return dt.tz_convert("UTC").tz_localize(None)
        return dt


class FeatureEngineer:
    """Merges all data sources into a single hourly feature matrix."""

    def build_feature_matrix(
        self,
        lmp_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        gen_mix_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        macro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge all data sources on hourly timestamps and engineer features.

        All lag/rolling features are strictly backward-looking to prevent
        data leakage. At prediction time t, only data from before t is used.

        Lag and rolling features are computed PER LOCATION to avoid
        cross-contamination between pricing nodes (e.g. CAISO has 3 nodes
        per hour, ERCOT has 4 hubs per hour).
        """

        # ── Prepare generation mix (pivot fuel types to columns) ──────────
        gen_wide = gen_mix_df.pivot_table(
            index="period", columns="fueltype", values="value", aggfunc="sum"
        ).add_suffix("_mwh")
        gen_wide.index = to_naive_utc(pd.to_datetime(gen_wide.index, utc=True))
        gen_wide = gen_wide.apply(pd.to_numeric, errors="coerce")

        # ── Prepare demand ────────────────────────────────────────────────
        demand_hourly = demand_df.copy()
        demand_hourly["period"] = pd.to_datetime(demand_hourly["period"], utc=True)
        demand_hourly["period"] = to_naive_utc(demand_hourly["period"])
        demand_hourly["demand"] = pd.to_numeric(demand_hourly["value"], errors="coerce")
        demand_hourly = (
            demand_hourly.groupby("period")["demand"].sum().to_frame()
        )

        # ── Prepare weather ───────────────────────────────────────────────
        weather_hourly = weather_df.copy()
        weather_hourly["time"] = pd.to_datetime(weather_hourly["time"])
        if weather_hourly["time"].dt.tz is not None:
            weather_hourly["time"] = to_naive_utc(weather_hourly["time"])
        weather_hourly = weather_hourly.drop(columns=["market"], errors="ignore")
        weather_hourly = weather_hourly.set_index("time")

        # ── Prepare macro (daily → forward-fill into hourly) ─────────────
        macro_daily = macro_df.copy()
        macro_daily["date"] = pd.to_datetime(macro_daily["date"])
        macro_daily = macro_daily.set_index("date")

        # ── Build base DataFrame from LMP data ───────────────────────────
        df = lmp_df.copy()
        df["hour"] = to_naive_utc(
            df["Time"].dt.tz_convert("UTC").dt.floor("h")
        )
        df = df.sort_values(["Location", "hour"]).reset_index(drop=True)

        # ── Join all sources on hour ──────────────────────────────────────
        df = df.set_index("hour")
        df = df.join(demand_hourly, how="left")
        df = df.join(gen_wide, how="left")
        df = df.join(weather_hourly, how="left")

        # Macro: join on date, then forward-fill daily values into hours
        df["_date"] = pd.to_datetime(df.index.date)
        df = df.join(macro_daily, on="_date", how="left")
        df = df.drop(columns=["_date"])
        df = df.ffill()
        df.index.name = "hour"
        df = df.reset_index()

        print(f"  Merged shape: {df.shape}")
        print(f"  Non-null LMP rows: {df['LMP'].notna().sum()}")
        print(f"  Non-null demand rows: {df['demand'].notna().sum()}")

        # ── Cyclical time features ────────────────────────────────────────
        df["hour_of_day"] = df["hour"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["day_of_week"] = df["hour"].dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month"] = df["hour"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # ── Lagged LMP features (per location, backward-looking only) ────
        grp = df.groupby("Location")["LMP"]
        for lag in [1, 2, 3, 4, 6, 12, 24, 48, 168]:
            df[f"lmp_lag_{lag}h"] = grp.shift(lag)

        # ── Rolling statistics (per location, shifted by 1 to avoid leak) ─
        shifted = grp.shift(1)
        for col_name, window in [
            ("lmp_rolling_mean_24h", 24),
            ("lmp_rolling_std_24h", 24),
            ("lmp_rolling_max_24h", 24),
            ("lmp_rolling_min_24h", 24),
            ("lmp_rolling_mean_168h", 168),
        ]:
            stat = col_name.split("_")[-1].rstrip("h").rstrip("0123456789")
            # rolling per location: group shifted series, apply rolling
            if "mean" in col_name:
                df[col_name] = shifted.groupby(df["Location"]).transform(
                    lambda x: x.rolling(window).mean()
                )
            elif "std" in col_name:
                df[col_name] = shifted.groupby(df["Location"]).transform(
                    lambda x: x.rolling(window).std()
                )
            elif "max" in col_name:
                df[col_name] = shifted.groupby(df["Location"]).transform(
                    lambda x: x.rolling(window).max()
                )
            elif "min" in col_name:
                df[col_name] = shifted.groupby(df["Location"]).transform(
                    lambda x: x.rolling(window).min()
                )

        # ── Price spread features (per location) ─────────────────────────
        shifted_1 = grp.shift(1)
        shifted_2 = grp.shift(2)
        shifted_25 = grp.shift(25)
        df["lmp_diff_1h"] = shifted_1 - shifted_2
        df["lmp_diff_24h"] = shifted_1 - shifted_25

        # ── Derived features (enable once gen_mix data is confirmed) ──────
        if "SUN_mwh" in df.columns and "demand" in df.columns:
            df["solar_penetration"] = df["SUN_mwh"] / df["demand"].clip(lower=1)

        if all(c in df.columns for c in ["demand", "SUN_mwh", "WND_mwh"]):
            df["net_load"] = df["demand"] - df["SUN_mwh"] - df["WND_mwh"]

        # ── Drop NaN rows from lagging (first ~168 rows per location) ────
        before = len(df)
        df = df.dropna()
        print(f"  Dropped {before - len(df)} rows with NaN (from lagging)")

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Build feature matrix from raw data"
    )
    parser.add_argument("--market", required=True, choices=["CAISO", "ERCOT", "both"])
    args = parser.parse_args()

    markets = ["CAISO", "ERCOT"] if args.market == "both" else [args.market]

    for market in markets:
        m = market.lower()
        print(f"\n{'='*60}")
        print(f"  Building feature matrix for {market}")
        print(f"{'='*60}")

        print("\nLoading raw data...")
        lmp_df = pd.read_parquet(RAW_DIR / f"{m}_lmps.parquet")
        demand_df = pd.read_parquet(RAW_DIR / f"{m}_demand.parquet")
        gen_mix_df = pd.read_parquet(RAW_DIR / f"{m}_gen_mix.parquet")
        weather_df = pd.read_parquet(RAW_DIR / f"{m}_weather.parquet")
        macro_df = pd.read_parquet(RAW_DIR / "macro_indicators.parquet")

        print(f"  LMPs:    {len(lmp_df):>7,} rows")
        print(f"  Demand:  {len(demand_df):>7,} rows")
        print(f"  Gen mix: {len(gen_mix_df):>7,} rows")
        print(f"  Weather: {len(weather_df):>7,} rows")
        print(f"  Macro:   {len(macro_df):>7,} rows")

        fe = FeatureEngineer()
        features = fe.build_feature_matrix(
            lmp_df, demand_df, gen_mix_df, weather_df, macro_df
        )

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DIR / f"{m}_features.parquet"
        features.to_parquet(output_path)

        print(f"\n✓ Saved {len(features):,} rows × {len(features.columns)} cols to {output_path}")
        print(f"  Date range: {features['hour'].min()} → {features['hour'].max()}")


if __name__ == "__main__":
    main()