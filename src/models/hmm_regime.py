import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib

# Hidden Markov Model for market regime classification
# Learns state from price returns, volatility, and renewable penetration 
# Outputs state probabilites that route to regime-specific models.

class RegimeDetector:

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=random_state,
        )
        self.regime_labels = {}  # Assigned after training based on state

    # Create observation matrix for HMM
    # Uses returns (not raw prices) + volatility + spread features
    def prepare_observations(self, df: pd.DataFrame) -> np.ndarray:
        obs = pd.DataFrame()
        obs["lmp_return"] = df["LMP"].pct_change()
        obs["lmp_volatility"] = df["LMP"].rolling(24).std()
        obs["lmp_level"] = df["LMP"]
        obs = obs.replace([np.inf, -np.inf], np.nan)
        obs = obs.dropna()
        return obs.values
    
    # Fit HMM on historical data
    def fit(self, df: pd.DataFrame):
        X = self.prepare_observations(df)
        self.model.fit(X)
        self._assign_regime_labels(X)
        return self
    
    # Predict current regime and return state probabilities
    # Returns: (most_likely_state, state_probabilites_array)
    def predict_regime(self, df: pd.DataFrame) -> tuple[int, np.ndarray]:
        X = self.prepare_observations(df)
        states = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return states[-1], probs[-1]
    
    # Label regimes by sorting state means
    def _assign_regime_labels(self, X: np.ndarray):
        """
        CAISO regimes (3-states):
            - States with lowest mean LMP & high solar -> "solar_surplus"
            - State with moderate mean -> "normal"
            - State with highest mean/volatility -> "peak_stress

        ERCOT regimes (3-states):
            - State with low mean/volatility -> "normal"
            - State with moderate elevation -> "high_demand"
            - State with extreme mean/high volatility -> "scarcity"
        """
        states = self.model.predict(X)
        # Sort by mean LMP (i = 2 in observation vector)
        state_means = {}
        for s in range(self.n_regimes):
            mask = states == s
            state_means[s] = X[mask, 2].mean()  # col 2 = lmp_level

        sorted_states = sorted(state_means, key=state_means.get)
        labels = ["low", "normal", "high"]
        self.regime_labels = {s: labels[i] for i, s in enumerate(sorted_states)}

    def save(self, path: str):
        joblib.dump({"model": self.model, "labels": self.regime_labels}, path)

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        data = joblib.load(path)
        detector = cls(n_regimes=data["model"].n_components)
        detector.model = data["model"]
        detector.regime_labels = data["labels"]
        return detector