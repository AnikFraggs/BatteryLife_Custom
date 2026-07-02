import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.naive_bayes import GaussianNB

# 1. MLP (Multi-Layer Perceptron)
class MLPModel(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim * 32, hidden), nn.GELU(), nn.Dropout(0.1), # Assuming window=32
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# 2. GRU (Gated Recurrent Unit)
class GRUModel(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)

# 3. Transformer
class TransformerModel(nn.Module):
    def __init__(self, in_dim, d=64, h=4, layers=2, max_len=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.pos  = nn.Parameter(torch.randn(1, max_len, d)*0.02)
        enc = nn.TransformerEncoderLayer(d, h, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.proj(x) + self.pos[:, :x.size(1)]
        x = self.tr(x)
        return self.head(x[:, -1]).squeeze(-1)

# 4. Naive Bayes + LSTM Hybrid
class NaiveBayesLSTM(nn.Module):
    def __init__(self, in_dim, n_classes=10, hidden=128, layers=2):
        super().__init__()
        self.n_classes = n_classes
        self.nb = None  # Will be fitted before training
        self.lstm = nn.LSTM(in_dim + n_classes, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))

    def fit_nb(self, x_sample):
        """Fits Gaussian NB on summary statistics of windows."""
        try:
            means = x_sample.mean(axis=1)
            stds = x_sample.std(axis=1)
            last_vals = x_sample[:, -1, :]
            summary_feats = np.concatenate([means, stds, last_vals], axis=1)
            
            # FIX: 'soh' is the 5th feature (index 4), not 5
            soh = x_sample[:, -1, 4] 
            quantiles = np.quantile(soh, np.linspace(0.1, 0.9, self.n_classes-1))
            labels = np.digitize(soh, quantiles)
            
            self.nb = GaussianNB().fit(summary_feats, labels)
        except Exception as e:
            print(f"Warning: fit_nb failed ({e}). Initializing fallback NB.")
            self.nb = GaussianNB()
        return self

    def forward(self, x):
        if self.nb is None:
            raise RuntimeError("Naive Bayes model not fitted. Call fit_nb() before forward pass.")
            
        B, T, F = x.shape
        # Extract summary for NB
        means = x.mean(dim=1).detach().cpu().numpy()
        stds = x.std(dim=1).detach().cpu().numpy()
        last_vals = x[:, -1, :].detach().cpu().numpy()
        summary_feats = np.concatenate([means, stds, last_vals], axis=1)
        
        cls = self.nb.predict(summary_feats)
        onehot = torch.zeros(B, self.n_classes, device=x.device)
        onehot.scatter_(1, torch.from_numpy(cls).long().unsqueeze(1).to(x.device), 1)
        onehot = onehot.unsqueeze(1).expand(-1, T, -1)
        
        x_in = torch.cat([x, onehot], dim=-1)
        out, _ = self.lstm(x_in)
        return self.head(out[:, -1]).squeeze(-1)