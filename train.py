import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from dataset import get_loaders, FEATURES
from model import MLPModel, GRUModel, TransformerModel, NaiveBayesLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"mlp": MLPModel, "gru": GRUModel, "transformer": TransformerModel, "nblstm": NaiveBayesLSTM}

def evaluate_loss(model, loader):
    model.eval()
    loss_fn = nn.HuberLoss()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            tot += loss_fn(model(x), y).item() * len(x)
            n += len(x)
    return tot / max(n, 1)

def train(model_name, chem, epochs=30, lr=1e-3):
    csv = f"data/{chem}.csv"
    tr, te, scaler = get_loaders(csv)
    in_dim = len(FEATURES)
    
    if model_name == "nblstm":
        model = NaiveBayesLSTM(in_dim)
    else:
        model = MODELS[model_name](in_dim)
    model = model.to(DEVICE)

    # Fit NB on a sample of training data before the loop starts
    if model_name == "nblstm":
        print("Fitting Naive Bayes classifier...")
        sample = next(iter(tr))[0].numpy()
        model.fit_nb(sample)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.HuberLoss()
    
    Path("ckpts").mkdir(exist_ok=True)
    best = float("inf")

    for ep in range(epochs):
        model.train()
        tot_loss = 0.0
        total_samples = 0
        
        for x, y in tqdm(tr, desc=f"{chem}/{model_name} ep{ep}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            
            tot_loss += loss.item() * len(x)
            total_samples += len(x)
            
        # FIX: Removed len(tr.dataset) to avoid attribute mismatch
        train_avg_loss = tot_loss / max(total_samples, 1)
        val_loss = evaluate_loss(model, te)
        print(f"  ep{ep} train {train_avg_loss:.2f}  val {val_loss:.2f}")
        
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), f"ckpts/{chem}_{model_name}.pt")
            if model_name == "nblstm":
                with open(f"ckpts/{chem}_{model_name}_nb.pkl", "wb") as f:
                    pickle.dump(model.nb, f)

    return best

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",  choices=list(MODELS), required=True)
    p.add_argument("--chem",   choices=["Li_ion","Na_ion","Zn_ion","CALB"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()
    train(a.model, a.chem, a.epochs)