"""
Phase 4: Dynamic Regime Rule Stability (V4)
===========================================
Evaluates the year-by-year stability of the optimal V4 Dynamic Rule.
"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "outputs" / "sim_labels_v4" / "v4_percentiles_labels.parquet"

def evaluate_rule(df, years, sp, op):
    print("\n" + "=" * 60)
    print(f" V4 STABILITY: Spread Pct >= {sp}%  &  OFI Pct <= {op}%")
    print("=" * 60)
    print(f"{'Year':<10} | {'Trades':<10} | {'Hit Rate':<15} | {'Profit Factor':<15}")
    print("-" * 60)
    
    total_trades = 0
    total_wins = 0
    total_losses = 0
    
    for y in years:
        mask = (df['year'] == y) & (df['spread_current_pct'] >= sp) & (df['abs_ofi_peak_pct'] <= op)
        subset = df[mask]
        trades = len(subset)
        if trades == 0:
            print(f"{y:<10} | {0:<10} | {'N/A':<15} | {'N/A':<15}")
            continue
            
        wins = subset['reversion'].sum()
        losses = trades - wins
        hr = (wins / trades) * 100
        pf = wins / max(losses, 1)
        
        total_trades += trades
        total_wins += wins
        total_losses += losses
        
        print(f"{y:<10} | {trades:<10} | {hr:>5.2f}%{' ':>9} | {pf:>5.2f}")
        
    print("-" * 60)
    if total_trades > 0:
        overall_hr = (total_wins / total_trades) * 100
        overall_pf = total_wins / max(total_losses, 1)
        print(f"{'OVERALL':<10} | {total_trades:<10} | {overall_hr:>5.2f}%{' ':>9} | {overall_pf:>5.2f}")
    print("=" * 60)

def main():
    df = pd.read_parquet(IN_PATH)
    df['year'] = pd.to_datetime(df['utc_day']).dt.year
    years = sorted(df['year'].unique())
    
    rules = [
        (95, 25),
        (95, 20),
        (97, 25),
        (97, 20),
        (99, 25)
    ]
    
    for sp, op in rules:
        evaluate_rule(df, years, sp, op)

if __name__ == "__main__":
    main()
