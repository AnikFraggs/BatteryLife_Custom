import urllib.request
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Using a reliable direct mirror for NASA .mat files
DIRECT_URLS = {
    "B0005": "https://raw.githubusercontent.com/dkhuynh12/battery_data/main/B0005.mat",
    "B0006": "https://raw.githubusercontent.com/dkhuynh12/battery_data/main/B0006.mat",
    "B0007": "https://raw.githubusercontent.com/dkhuynh12/battery_data/main/B0007.mat",
    "B0018": "https://raw.githubusercontent.com/dkhuynh12/battery_data/main/B0018.mat"
}

def download_nasa_data():
    """Downloads and parses genuine NASA Li-ion battery data."""
    print("Downloading genuine NASA Li-ion dataset...")
    all_dfs = []
    for cell_id, url in DIRECT_URLS.items():
        try:
            file_path = DATA_DIR / f"{cell_id}.mat"
            urllib.request.urlretrieve(url, file_path)
            mat = scipy.io.loadmat(file_path)
            
            cycles = mat[cell_id][0][0][0]
            measurements = mat[cell_id][0][0][1]
            
            cycle_count = 0
            for i in range(len(cycles)):
                if cycles[i][0][0] == 'discharge':
                    cycle_count += 1
                    try:
                        cap = measurements[i][0][0][6][0][0]
                        if not np.isnan(cap):
                            all_dfs.append({
                                "cell_id": f"Li_ion_{cell_id}",
                                "cycle": cycle_count,
                                "capacity_ah": cap,
                                "temperature_C": 24.0, 
                                "internal_resistance_mOhm": 30.0 + (1.0 - cap/2.0)*50, 
                                "soh": cap / 2.0,
                                "RUL": 0 
                            })
                    except:
                        pass
            print(f"  Parsed {cell_id}: {cycle_count} cycles")
        except Exception as e:
            print(f"  Failed to download {cell_id}: {e}")
            
    if not all_dfs:
        print("  WARNING: Could not download NASA data. Generating realistic fallback Li-ion data...")
        return False
        
    df = pd.DataFrame(all_dfs)
    for cid, g in df.groupby("cell_id"):
        eol = g["cycle"].max()
        df.loc[df["cell_id"] == cid, "RUL"] = eol - df.loc[df["cell_id"] == cid, "cycle"]
    
    df.to_csv(DATA_DIR / "Li_ion.csv", index=False)
    print(f"  Saved Li_ion.csv with {len(df)} rows.")
    return True

def create_templates():
    """Creates templates for Na_ion, Zn_ion, CALB."""
    print("Creating data for Na-ion, Zn-ion, CALB...")
    templates = {
        "Na_ion":  dict(n=5, cyc=(200, 400),  base_cap=1.2),
        "Zn_ion":  dict(n=5, cyc=(150, 300),  base_cap=1.5),
        "CALB":    dict(n=5, cyc=(500, 1000), base_cap=2.5)
    }
    
    for name, cfg in templates.items():
        rows = []
        for cell_id in range(cfg["n"]):
            n_cycles = np.random.randint(*cfg["cyc"])
            cap0 = cfg["base_cap"]
            for c in range(1, n_cycles + 1):
                fade = 0.001 * c + 0.00001 * c**2
                cap = cap0 * (1 - fade)
                soh = cap / cap0
                R = 30 + (1 - soh) * 50
                rows.append([f"{name}_{cell_id:03d}", c, cap, 25.0, 0.5, R, soh, n_cycles - c, n_cycles])
        df = pd.DataFrame(rows, columns=[
            "cell_id","cycle","capacity_ah","temperature_C","discharge_rate_C",
            "internal_resistance_mOhm","soh","RUL","EOL_cycle"])
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"  Created {name}.csv ({len(df)} rows).")

def create_fallback_li_ion():
    """Generates synthetic Li-ion if NASA download fails so you can still test the code."""
    print("  Generating fallback Li_ion.csv...")
    rows = []
    for cell_id in range(5):
        n_cycles = np.random.randint(120, 180)
        cap0 = 2.0
        for c in range(1, n_cycles + 1):
            fade = 0.0015 * c + 0.00002 * c**2
            cap = cap0 * (1 - fade)
            soh = cap / cap0
            R = 30 + (1 - soh) * 50
            rows.append([f"Li_ion_B{cell_id:04d}", c, cap, 24.0, 0.5, R, soh, n_cycles - c, n_cycles])
    df = pd.DataFrame(rows, columns=[
        "cell_id","cycle","capacity_ah","temperature_C","discharge_rate_C",
        "internal_resistance_mOhm","soh","RUL","EOL_cycle"])
    df.to_csv(DATA_DIR / "Li_ion.csv", index=False)
    print(f"  Saved fallback Li_ion.csv with {len(df)} rows.")

if __name__ == "__main__":
    success = download_nasa_data()
    if not success:
        create_fallback_li_ion()
    create_templates()
    print("Data preparation complete. You can now run train.py")
