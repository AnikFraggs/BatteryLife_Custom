import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

FEATURES = ["cycle", "capacity_ah", "temperature_C", "internal_resistance_mOhm", "soh"]
TARGET = "RUL"
WINDOW = 32

class BatteryWindowDataset(Dataset):
    def __init__(self, df, scaler=None, fit_scaler=False):
        self.X, self.y = [], []
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(df[FEATURES].values)
            fit_scaler = True
        self.scaler = scaler
        
        df = df.copy()
        df[FEATURES] = scaler.transform(df[FEATURES].values)

        for cid, g in df.groupby("cell_id"):
            g = g.sort_values("cycle").reset_index(drop=True)
            arr = g[FEATURES].values.astype(np.float32)
            rul = g[TARGET].values.astype(np.float32)
            for i in range(len(g) - WINDOW):
                self.X.append(arr[i:i+WINDOW])
                self.y.append(rul[i+WINDOW])
                
        self.X = np.stack(self.X)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

def get_loaders(csv_path, batch=64):
    df = pd.read_csv(csv_path)
    cells = df["cell_id"].unique()
    np.random.shuffle(cells)
    
    n_test = max(1, int(len(cells) * 0.2))
    test_cells = cells[:n_test]
    
    train_df = df[~df["cell_id"].isin(test_cells)]
    test_df  = df[df["cell_id"].isin(test_cells)]

    train_ds = BatteryWindowDataset(train_df, fit_scaler=True)
    test_ds  = BatteryWindowDataset(test_df, scaler=train_ds.scaler)
    
    tr = DataLoader(train_ds, batch_size=batch, shuffle=True)
    te = DataLoader(test_ds, batch_size=batch, shuffle=False)
    return tr, te, train_ds.scaler