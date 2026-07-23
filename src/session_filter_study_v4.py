"""
Phase 5: Session Context Filter Study (V4)
==========================================
Evaluates the impact of restricting the V4 Dynamic Regime rule
to specific UTC trading sessions, explicitly excluding the 
2020 Black Swan (March 24) data.
"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "outputs" / "sim_labels_v4" / "v4_percentiles_labels.parquet"

def print_metrics(subset, name):
    trades = len(subset)
    if trades == 0:
        print(f"{name:<25} | 0 trades")
        return
    wins = subset['reversion'].sum()
    hr = (wins / trades) * 100
    pf = wins / max(trades - wins, 1)
    print(f"{name:<25} | {trades:<6} | {hr:>5.2f}% | {pf:>5.2f}")

def main():
    df = pd.read_parquet(IN_PATH)
    
    # Extract Datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Exclude 2020 completely
    df = df[df['datetime'].dt.year > 2020].copy()
    df['hour'] = df['datetime'].dt.hour
    
    # Base Dynamic Rule
    mask = (df['spread_current_pct'] >= 95) & (df['abs_ofi_peak_pct'] <= 20)
    base_df = df[mask].copy()
    
    print("\n" + "=" * 60)
    print(" SESSION FILTER STUDY (Excluding 2020)")
    print(" Rule: Spread Pct >= 95%  &  OFI Pct <= 20%")
    print("=" * 60)
    
    print_metrics(base_df, "Rule A (All Hours)")
    
    # Rule B: London Open (08:00 - 11:00 UTC)
    rule_b = base_df[base_df['hour'].isin([8, 9, 10, 11])]
    print_metrics(rule_b, "Rule B (London 08-11)")
    
    # Rule C: London + NY (08:00 - 14:00 UTC)
    rule_c = base_df[base_df['hour'].isin([8, 9, 10, 11, 12, 13, 14])]
    print_metrics(rule_c, "Rule C (Lon+NY 08-14)")
    
    # Rule D: Exclude Weak Hours (03, 15, 18, 22, 23)
    weak_hours = [3, 15, 18, 22, 23]
    rule_d = base_df[~base_df['hour'].isin(weak_hours)]
    print_metrics(rule_d, "Rule D (Exclude Weak)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
