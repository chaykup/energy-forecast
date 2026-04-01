import xgboost as xgb
import numpy as np
import pandas as pd
import joblib

# XGBoost model trained on data from a specific market regime
class RegimeXGBoost:

    # Features to exclude from training (target, metadata, etc.)
    EXCLUDE_COLS = ["LMP", "Time", "hour", "iso", "Location", "Market",
                "Location Name", "Location Type",
                "Interval Start", "Interval End"]

    def __init__(self, regime_id: int, market: str, **xgb_params):
        self.regime_id = regime_id
        self.market = market
        self.params = {
            "n_estimators": 500,
            "max_depth": 6, 
            "learning_rate": 0.01, 
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            **xgb_params,
        }
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_cols = None

    # Train on rows belonging to this regime
    # regime_states: array of regime IDs aligned with df index
    def fit(self, df: pd.DataFrame, regime_states: np.ndarray):
        mask = regime_states == self.regime_id
        regime_df = df[mask].copy()
        self.feature_cols = [c for c in regime_df.select_dtypes(include=["number", "bool"]).columns
                            if c not in ("LMP", "regime")]
        X = regime_df[self.feature_cols]
        y = regime_df["LMP"]
        self.model.fit(X, y)
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols]
        return self.model.predict(X)
    
    # Compute residuals for LSTM training
    def get_residuals(self, df: pd.DataFrame, regime_states: np.ndarray) -> pd.Series:
        mask = regime_states == self.regime_id
        regime_df = df[mask]
        preds = self.predict(regime_df)
        residuals = regime_df["LMP"].values - preds
        return pd.Series(residuals, index=regime_df.index)
    
    def save(self, path: str):
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols, "regime_id": self.regime_id, "market": self.market}, path)

    @classmethod
    def load(cls, path: str) -> "RegimeXGBoost":
        data = joblib.load(path)
        obj = cls(regime_id=data["regime_id"], market=data["market"])
        obj.model = data["model"]
        obj.feature_cols = data ["feature_cols"]
        return obj