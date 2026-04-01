import pandas as pd
from nixtla import NixtlaClient
from src.utils.config import NIXTLA_API_KEY


# TimeGPT benchmark for LMP forecasting
# Runs zero-shot and fine-tuned variants for comparison


class TimeGPTBaseline:

    def __init__(self):
        self.client = NixtlaClient(api_key=NIXTLA_API_KEY)

    # Zero-shot forecast - no training
    #   - TimeGPT without exposure to specific data
    # df must have columns: hour (naive UTC datetime), LMP
    def forecast_zero_shot(
        self, df: pd.DataFrame, horizon: int = 24, market: str = "CAISO"
    ) -> pd.DataFrame:
        input_df = self._prepare_input(df, market)
        fcst = self.client.forecast(
            df=input_df,
            h=horizon,
            freq="h",
            level=[80, 90],
            time_col="ds",
            target_col="y",
            id_col="unique_id",
        )
        fcst["model"] = "timegpt_zero_shot"
        return fcst

    # Fine-tuned forecast - adapts pretrained model to your data
    # exog_df: optional DataFrame with exogenous variables aligned by timestamp
    #   - temperature, solar irradiance, demand, etc.
    # Must include future values for forecast horizon
    def forecast_finetuned(
        self,
        df: pd.DataFrame,
        horizon: int = 24,
        market: str = "CAISO",
        finetune_steps: int = 30,
        exog_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        input_df = self._prepare_input(df, market)
        kwargs = {
            "df": input_df,
            "h": horizon,
            "freq": "h",
            "level": [80, 90],
            "finetune_steps": finetune_steps,
            "finetune_loss": "mae",
            "time_col": "ds",
            "target_col": "y",
            "id_col": "unique_id",
        }

        # Add exogenous variables if provided
        if exog_df is not None:
            kwargs["X_df"] = exog_df

        # Use long-horizon model if forecasting > 24 steps
        if horizon > 24:
            kwargs["model"] = "timegpt-1-long-horizon"

        fcst = self.client.forecast(**kwargs)
        fcst["model"] = "timegpt_finetuned"
        return fcst

    # Convert LMP DataFrame to Nixtla's expected format
    # Uses "hour" column (naive UTC) — guaranteed gapless after reindex
    def _prepare_input(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        input_df = df[["hour", "LMP", "Location"]].copy()
        input_df = input_df.rename(columns={"hour": "ds", "LMP": "y", "Location": "unique_id"})
        input_df["ds"] = pd.to_datetime(input_df["ds"])
        input_df = input_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # Gapless reindex per node
        frames = []
        for uid, group in input_df.groupby("unique_id"):
            full_range = pd.date_range(start=group["ds"].min(), end=group["ds"].max(), freq="h")
            reindexed = (
                group.set_index("ds")
                .reindex(full_range)
                .rename_axis("ds")
                .reset_index()
            )
            reindexed["unique_id"] = uid
            reindexed["y"] = reindexed["y"].interpolate(method="linear")
            frames.append(reindexed)
        return pd.concat(frames, ignore_index=True)

    # Walk forward cross-validation
    # Returns DataFrame with actual vs predicted for each window
    def cross_validate(
        self,
        df: pd.DataFrame,
        horizon: int = 24,
        market: str = "CAISO",
        n_windows: int = 5,
        finetune_steps: int = 30,
    ) -> pd.DataFrame:
        input_df = self._prepare_input(df, market)
        cv_results = self.client.cross_validation(
            df=input_df,
            h=horizon,
            freq="h",
            n_windows=n_windows,
            finetune_steps=finetune_steps,
            finetune_loss="mae",
            time_col="ds",
            target_col="y",
            id_col="unique_id",
        )
        return cv_results