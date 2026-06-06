import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    parquet_path = BASE_DIR / "outputs" / "sim_labels_v4" / "v4_percentiles_labels.parquet"
    print(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Filter for 2026 out-of-sample
    df_2026 = df[df['year'] == 2026].copy()
    
    # Apply execution deduplication filter (gap-fill protection)
    df_2026 = df_2026.drop_duplicates(subset=['timestamp'], keep='first')
    
    # Apply the 99/3 Structural Rule
    mask = (df_2026['spread_current_pct'] >= 99) & (df_2026['abs_ofi_peak_pct'] <= 3)
    trades = df_2026[mask].copy()
    
    print(f"Total 2026 Dukascopy Trades (99/3): {len(trades)}")
    
    for rr in [1.0, 1.5, 2.0, 3.0]:
        wins = (trades['reversion_depth'] >= rr).sum()
        total = len(trades)
        losses = total - wins
        wr = wins / total if total > 0 else 0
        pf = (wins * rr) / (losses * 1.0 + 1e-8)
        pnl = (wins * rr) - losses
        
        print(f"\n--- {rr}R TARGET ---")
        print(f" Trades: {total}")
        print(f" Win Rate: {wr*100:.2f}%")
        print(f" Profit Factor: {pf:.2f}")
        print(f" Total PnL: {pnl:+.2f}R")

if __name__ == "__main__":
    main()
