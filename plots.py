import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CHEMS  = ["Li_ion","Na_ion","Zn_ion","CALB"]
COLORS = {"Li_ion":"#1f77b4","Na_ion":"#ff7f0e","Zn_ion":"#2ca02c","CALB":"#d62728"}
MODELS = ["mlp","gru","transformer","nblstm"]

# 1. Battery Life Distribution
fig, ax = plt.subplots(figsize=(8,5))
for c in CHEMS:
    try:
        df = pd.read_csv(f"data/{c}.csv")
        life = df.groupby("cell_id")["cycle"].max()
        ax.hist(life, bins=15, alpha=0.55, label=c, color=COLORS[c])
    except: pass
ax.set_xlabel("End-of-life cycle"); ax.set_ylabel("# cells")
ax.set_title("1. Battery Life Distribution")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig_life_distribution.png", dpi=150); plt.close()

# 2. Error Distribution
res = pd.read_csv("results.csv")
res["err"] = res["y_pred"] - res["y_true"]
fig, axes = plt.subplots(1, 4, figsize=(18,4), sharey=True)
for ax, chem in zip(axes, CHEMS):
    sub = res[res.chemistry==chem]
    for m in MODELS:
        data = sub[sub.model==m]["err"]
        if len(data) > 0:
            ax.hist(data, bins=40, alpha=0.5, label=m)
    ax.set_title(f"{chem} - Prediction Error"); ax.set_xlabel("pred - true (cycles)")
    ax.legend(); ax.grid(alpha=0.3); ax.axvline(0,c="k",lw=0.7)
plt.tight_layout(); plt.savefig("fig_error_distribution.png", dpi=150); plt.close()

# 3. SOH Trajectories
fig, axes = plt.subplots(1, 4, figsize=(18,4))
for ax, chem in zip(axes, CHEMS):
    try:
        df = pd.read_csv(f"data/{chem}.csv")
        for cid, g in df.groupby("cell_id"):
            g = g.sort_values("cycle")
            ax.plot(g["cycle"], g["soh"], color=COLORS[chem], alpha=0.25, lw=0.6)
        mean = df.groupby("cycle")["soh"].mean()
        ax.plot(mean.index, mean.values, color="black", lw=2, label="mean")
        ax.axhline(0.8, ls="--", c="gray", lw=1, label="EOL (80% SoH)")
    except: pass
    ax.set_title(f"{chem} SoH Trajectories")
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig_soh_trajectories.png", dpi=150); plt.close()
print("Saved 3 plots: fig_life_distribution.png, fig_error_distribution.png, fig_soh_trajectories.png")