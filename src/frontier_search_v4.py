"""
Phase 4: Percentile Frontier Grid Search (V4)
=============================================
Runs an interaction grid search across Spread and OFI percentiles.
"""

import pandas as pd
from pathlib import Path
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "outputs" / "sim_labels_v4" / "v4_percentiles_labels.parquet"
OUT_DIR = BASE_DIR / "outputs" / "experiments"

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    df = pd.read_parquet(IN_PATH)
    
    spread_pcts = [75, 80, 85, 90, 95, 97, 99]
    ofi_pcts = [50, 40, 30, 25, 20, 15, 10, 5]
    
    results = []
    
    for sp in spread_pcts:
        for op in ofi_pcts:
            mask = (df['spread_current_pct'] >= sp) & (df['abs_ofi_peak_pct'] <= op)
            subset = df[mask]
            trades = len(subset)
            if trades == 0:
                continue
            
            wins = subset['reversion'].sum()
            losses = trades - wins
            hr = (wins / trades) * 100
            pf = wins / max(losses, 1)
            
            results.append({
                "Spread_Pct": sp,
                "OFI_Pct": op,
                "Trades": trades,
                "Hit_Rate": hr,
                "PF": pf
            })
            
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by=["Spread_Pct", "OFI_Pct"], ascending=[True, False])
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT_DIR / "v4_frontier_grid.csv", index=False)
    
    # Print Markdown table
    print("| Spread Pct | OFI Pct | Trades | Hit Rate | Profit Factor |")
    print("| :--- | :--- | :---: | :---: | :---: |")
    for _, row in res_df.iterrows():
        print(f"| >={int(row['Spread_Pct'])}% | <={int(row['OFI_Pct'])}% | {int(row['Trades'])} | {row['Hit_Rate']:.2f}% | {row['PF']:.2f} |")

if __name__ == "__main__":
    main()
