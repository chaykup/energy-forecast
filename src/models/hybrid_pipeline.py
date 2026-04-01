import numpy as np
import pandas as pd
from src.models.hmm_regime import RegimeDetector
from src.models.xgboost_model import RegimeXGBoost
from src.models.lstm_model import RegimeLSTM

# Pipeline: regime detection -> route to XGBoost + LSTM -> forecast
# One instance per market (CAISO or ERCOT)

class HybridPipeline:

    def __init__(self, market: str, n_regimes: int = 3):
        self.market = market
        self.n_regimes = n_regimes
        self.regime_detector = RegimeDetector(n_regimes=n_regimes)
        self.xgb_models = {}    # {regime_id: RagimeXGBoost}
        self.lstm_models = {}   # {regime_id: RegimeLSTM}

    # Full training pipeline
    def train(self, df: pd.DataFrame):
        print(f"\n{'=' * 60}")
        print(f"Training pipeline for {self.market}")
        print(f"{'=' * 60}")

        # Fit HMM
        print("\nFitting HMM regime detector...")
        self.regime_detector.fit(df)
        X_hmm = self.regime_detector.prepare_observations(df)
        states = self.regime_detector.model.predict(X_hmm)

        # Align states with df (HMM drops NaN rows from prepare_observations)
        df_aligned = df.iloc[-len(states):].copy()
        df_aligned["regime"] = states

        for regime_id in range(self.n_regimes):
            n_samples = (states == regime_id).sum()
            label = self.regime_detector.regime_labels.get(regime_id, "unknown")
            print(f"    Regime {regime_id} ({label}):   {n_samples} samples")

        # Train per-regime XGBoost
        print("\nTraining regime-conditional XGBoost models...")
        for regime_id in range(self.n_regimes):
            print(f"    Training XGBoost for regime {regime_id}...")
            xgb_model = RegimeXGBoost(regime_id=regime_id, market=self.market)
            xgb_model.fit(df_aligned, states)
            self.xgb_models[regime_id] = xgb_model

        # Train per-regime LSTM on residuals
        print("\nTraining regime-conditional LSTM residual models...")
        for regime_id in range(self.n_regimes):
            print(f"    Training LSTM for regime {regime_id}...")
            residuals = self.xgb_models[regime_id].get_residuals(df_aligned, states)
            lstm_model = RegimeLSTM(regime_id=regime_id, market=self.market)
            lstm_model.fit(residuals.values)
            self.lstm_models[regime_id] = lstm_model
        
        print(f"\nTraining complete for {self.market}.")

    # Run inference on current data
    # Returns dict with forecast, regime info, and recommendation
    def predict(self, df: pd.DataFrame) -> dict:

        # Detect current regime
        current_regime, regime_probs = self.regime_detector.predict_regime(df)
        regime_label = self.regime_detector.regime_labels.get(current_regime, "unknown")

        # Get XGBoost prediction
        xgb_pred = self.xgb_models[current_regime].predict(df.iloc[[-1]])[0]

        # Get LSTM residual correction
        residuals = self.xgb_models[current_regime].get_residuals(
            df, self.regime_detector.model.predict(
                self.regime_detector.prepare_observations(df)
            )
        )
        lstm_correction = self.lstm_models[current_regime].predict(residuals.values)

        # Final forecast
        forecast_lmp = xgb_pred + lstm_correction

        # Battery recommendation
        current_lmp = df["LMP"].iloc[-1]
        recommendation = self._battery_recommendation(
            current_lmp, forecast_lmp, regime_label
        )

        return {
            "market": self.market,
            "current_lmp": float(current_lmp),
            "forecast_lmp": float(forecast_lmp),
            "xgb_component": float(xgb_pred),
            "lstm_correction": float(lstm_correction), 
            "current_regime": int(current_regime),
            "regime_label": regime_label,
            "regime_probabilities": regime_probs.tolist(),
            "recommendation": recommendation,
        }
    
    def _battery_recommendation(self, current_lmp: float, forecast_lmp: float, regime: str) -> dict:
        """
        Generate charge/hold/discharge recommendation
        Logic:
            - If forecast >> current: CHARGE (buy cheap, sell expensive)
            - If forecast << current: DISCHARGE (sell at high)
            - If ~flat: HOLD
            - In spike regime: DISCHARGE aggressively
            - In surplus regime: CHARGE aggressively
        """
        spread = forecast_lmp - current_lmp
        pct_change = spread / (abs(current_lmp) + 1e-6)

        if regime in ("high", "scarcity"):
            if pct_change > 0.05:
                action, confidence = "HOLD", "medium"
            else:
                action, confidence = "DISCHARGE", "high"
        elif regime in ("low", "solar_surplus"):
            if current_lmp < 10: # Very low or negative
                action, confidence = "CHARGE", "high"
            else:
                action, confidence = "CHARGE", "medium"
        else: # Normal regime
            if pct_change > 0.10:
                action, confidence = "CHARGE", "medium"
            elif pct_change < -0.10:
                action, confidence = "DISCHARGE", "medium"
            else:
                action, confidence = "HOLD", "low"

        return {
            "action": action,
            "confidence": confidence,
            "spread_dollar": float(spread),
            "spread_pct": float(pct_change),
        }
    
    # Save model artifacts
    def save(self, base_path: str):
        self.regime_detector.save(f"{base_path}/{self.market}_hmm.joblib")
        for rid, model in self.xgb_models.items():
            model.save(f"{base_path}/{self.market}_xgb_regime{rid}.joblib")
        for rid, model in self.lstm_models.items():
            model.save(f"{base_path}/{self.market}_lstm_regime{rid}.pt")

    # Load all model artifacts
    @classmethod
    def load(cls, base_path: str, market: str, n_regimes: int = 3) -> "HybridPipeline":
        pipeline = cls(market=market, n_regimes=n_regimes)
        pipeline.regime_detector = RegimeDetector.load(f"{base_path}/{market}_hmm.joblib")
        for rid in range(n_regimes):
            pipeline.xgb_models[rid] = RegimeXGBoost.load(f"{base_path}/{market}_xgb_regime{rid}.joblib")
            pipeline.lstm_models[rid] = RegimeLSTM.load(f"{base_path}/{market}_lstm_regime{rid}.pt")
        return pipeline