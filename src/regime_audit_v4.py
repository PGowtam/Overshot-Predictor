"""
Phase 5: Regime Audit (V4)
==========================
Analyzes the distribution of 95/20 dynamic rule signals across
UTC hours and days of the week to detect structural liquidity vacuums.
"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "outputs" / "sim_labels_v4" / "v4_percentiles_labels.parquet"

def main():
    df = pd.read_parquet(IN_PATH)
    
    mask = (df['spread_current_pct'] >= 95) & (df['abs_ofi_peak_pct'] <= 20)
    subset = df[mask].copy()
    
    subset['datetime'] = pd.to_datetime(subset['timestamp'], unit='ms', utc=True)
    subset['hour'] = subset['datetime'].dt.hour
    subset['day_of_week'] = subset['datetime'].dt.day_name()
    
    total_trades = len(subset)
    print("=" * 60)
    print(f" REGIME AUDIT: Spread >= 95% & OFI <= 20% (Total: {total_trades})")
    print("=" * 60 + "\n")
    
    print("=== HOUR OF DAY AUDIT (UTC) ===")
    print(f"{'Hour':<6} | {'Trades':<8} | {'Hit Rate':<10} | {'Profit Factor':<15}")
    print("-" * 50)
    for h in range(24):
        h_df = subset[subset['hour'] == h]
        trades = len(h_df)
        if trades == 0:
            continue
        wins = h_df['reversion'].sum()
        losses = trades - wins
        hr = (wins / trades) * 100
        pf = wins / max(losses, 1)
        print(f"{h:02d}:00  | {trades:<8} | {hr:>5.2f}%    | {pf:>5.2f}")
        
    print("\n=== DAY OF WEEK AUDIT (UTC) ===")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    print(f"{'Day':<10} | {'Trades':<8} | {'Hit Rate':<10} | {'Profit Factor':<15}")
    print("-" * 50)
    for d in days:
        d_df = subset[subset['day_of_week'] == d]
        trades = len(d_df)
        if trades == 0:
            continue
        wins = d_df['reversion'].sum()
        losses = trades - wins
        hr = (wins / trades) * 100
        pf = wins / max(losses, 1)
        print(f"{d:<10} | {trades:<8} | {hr:>5.2f}%    | {pf:>5.2f}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
