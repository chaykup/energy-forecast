import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Sliding window dataset of XGBoost residuals
class ResidualDataset(Dataset):

    def __init__(self, residuals: np.ndarray, seq_len: int = 24):
        self.seq_len = seq_len
        self.data = torch.FloatTensor(residuals)

    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]   # input window
        y = self.data[idx + self.seq_len]         # next residual
        return x.unsqueeze(-1), y                 # (seq_len, 1), scalar
    
# LSTM that predicts the next XGBoost residual from a window of past residuals
class ResidualLSTM(nn.Module):

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]    # last timestep
        return self.fc(last_hidden).squeeze(-1)

# Wrapper for training and inference of per-regime LSTM    
class RegimeLSTM:
    def __init__(self, regime_id: int, market: str, seq_len: int = 24, hidden_size: int = 64, num_layers: int = 2, lr: float = 1e-3, epochs: int = 50, batch_size: int = 32):
        self.regime_id = regime_id
        self.market = market
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResidualLSTM(
            hidden_size=hidden_size, num_layers=num_layers
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.residual_mean = 0.0
        self.residual_std = 1.0

    # Train LSTM on XGBoost residual sequence
    def fit(self, residuals: np.ndarray):
        # Normalize residuals
        self.residual_mean = residuals.mean()
        self.residual_std = residuals.std() + 1e-8
        normed = (residuals - self.residual_mean / self.residual_std)

        dataset = ResidualDataset(normed, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"    LSTM [{self.market}/regime_{self.regime_id}] "
                      f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.6f}")
                
    # Predict next residual from a window of recent residuals
    # recent_residuals: array of length >= seq_len
    def predict(self, recent_residuals: np.ndarray) -> float:
        self.model.eval()
        normed = (recent_residuals[-self.seq_len:] - self.residual_mean) / self.residual_std
        x = torch.FloatTensor(normed).unsqueeze(0).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            pred_normed = self.model(x).item()
        return pred_normed * self.residual_std + self.residual_mean
    
    def save(self, path:str):
        torch.save({
            "model_state": self.model.state_dict(),
            "regime_id": self.regime_id,
            "market": self.market,
            "seq_len": self.seq_len,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
        }, path)

    @classmethod
    def load(cls, path: str) -> "RegimeLSTM":
        data = torch.load(path, weights_only=False)
        obj = cls(regime_id=data["regime_id"], market=data["market"], seq_len=data["seq_len"])
        obj.model.load_state_dict(data["model_state"])
        obj.residual_mean = data["residual_mean"]
        obj.residual_std = data["residual_std"]
        return obj