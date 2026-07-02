import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from dataset import get_loaders, FEATURES
from model import MLPModel, GRUModel, TransformerModel, NaiveBayesLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHEMS  = ["Li_ion","Na_ion","Zn_ion","CALB"]
MODELS = ["mlp","gru","transformer","nblstm"]

def load_model(name, in_dim, ckpt, chem):
    if name == "nblstm":
        m = NaiveBayesLSTM(in_dim)
        with open(f"ckpts/{chem}_{name}_nb.pkl","rb") as f:
            m.nb = pickle.load(f)
    else:
        m = {"mlp":MLPModel,"gru":GRUModel,"transformer":TransformerModel}[name](in_dim)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return m.to(DEVICE).eval()

def predict(model, loader):
    ys, ps = [], []
    with torch.no_grad():
        for x,y in loader:
            ys.append(y.numpy())
            ps.append(model(x.to(DEVICE)).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

if __name__ == "__main__":
    rows = []
    for chem in CHEMS:
        csv = f"data/{chem}.csv"
        _, te, _ = get_loaders(csv)
        for mname in MODELS:
            ckpt = f"ckpts/{chem}_{mname}.pt"
            if not Path(ckpt).exists():
                print(f"skip {chem}/{mname} (no ckpt)"); continue
            m = load_model(mname, len(FEATURES), ckpt, chem)
            y, p = predict(m, te)
            
            for yt, yp in zip(y, p):
                rows.append({"chemistry": chem, "model": mname, "y_true": float(yt), "y_pred": float(yp)})
                
            rmse = np.sqrt(np.mean((y-p)**2))
            mae  = np.mean(np.abs(y-p))
            print(f"{chem:8s} {mname:11s} RMSE={rmse:6.1f}  MAE={mae:6.1f}")
            
    pd.DataFrame(rows).to_csv("results.csv", index=False)
    print("wrote results.csv")