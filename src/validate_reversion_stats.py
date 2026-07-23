"""
Phase 3: Statistical Validation for Reversion/Exhaustion
========================================================
Validates the structural baseline of the V3 labels and computes
1D and 2D interaction tables for the new Exhaustion features.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
V3_LBL_DIR = BASE_DIR / "outputs" / "sim_labels_v3"

def main():
    print("Loading V3 labels...")
    files = glob.glob(str(V3_LBL_DIR / "v3_labels_*.parquet"))
    if not files:
        print("No V3 labels found!")
        return
        
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"\nTotal Samples: {len(df)}")
    
    # Derive targets
    df['y_reversion'] = df['path_class'].isin([1, 2]).astype(int)
    df['y_full_reversal'] = (df['path_class'] == 2).astype(int)
    
    # Drop NaNs
    df = df.dropna(subset=['y_reversion', 'ofi_peak', 'vel_peak', 'spread_current', 'absorption_index', 'wick_ratio'])
    
    print("\n--- BASELINE FREQUENCIES ---")
    rev_rate = df['y_reversion'].mean() * 100
    full_rev_rate = df['y_full_reversal'].mean() * 100
    print(f"Baseline Reversion Rate: {rev_rate:.2f}%")
    print(f"Baseline Full Reversal Rate: {full_rev_rate:.2f}%")
    print(f"Average Reversion Depth: {df['reversion_depth'].mean():.2f} bricks")
    
    # Absolute OFI peak
    df['abs_ofi_peak'] = df['ofi_peak'].abs()
    
    features = [
        ('abs_ofi_peak', "Absolute OFI Peak (High = High Flow)"),
        ('vel_peak', "Velocity Peak (High = Fast Momentum)"),
        ('spread_current', "Current Spread (High = Liquidity Withdrawal)"),
        ('absorption_index', "Absorption Index (OFI / Velocity)"),
        ('wick_ratio', "Wick Ratio (Rejection length)"),
        ('ofi_slope', "OFI Slope (Momentum Acceleration)"),
    ]
    
    for feat, desc in features:
        print(f"\n--- 1D Table: {desc} ---")
        try:
            df[f'{feat}_q'] = pd.qcut(df[feat], 4, duplicates='drop')
            res = df.groupby(f'{feat}_q', observed=True)[['y_reversion', 'y_full_reversal', 'reversion_depth']].mean()
            res['y_reversion'] *= 100
            res['y_full_reversal'] *= 100
            res['count'] = df.groupby(f'{feat}_q', observed=True).size()
            print(res.to_string(float_format=lambda x: f"{x:.2f}"))
        except Exception as e:
            print(f"Could not bin {feat}: {e}")
            
    print("\n--- 2D INTERACTION: OFI Peak vs Velocity Peak ---")
    try:
        df['ofi_class'] = pd.qcut(df['abs_ofi_peak'], 3, labels=['Low OFI', 'Med OFI', 'High OFI'])
        df['vel_class'] = pd.qcut(df['vel_peak'], 3, labels=['Low Vel', 'Med Vel', 'High Vel'])
        pivot = df.pivot_table(
            index='ofi_class', 
            columns='vel_class', 
            values='y_reversion', 
            aggfunc='mean',
            observed=True
        ) * 100
        print("Reversion Rate (%):")
        print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))
    except Exception as e:
        print(f"Interaction failed: {e}")
        
    print("\n--- 2D INTERACTION: Absorption vs Wick Ratio ---")
    try:
        df['abs_class'] = pd.qcut(df['absorption_index'], 3, labels=['Low Abs', 'Med Abs', 'High Abs'])
        df['wick_class'] = pd.qcut(df['wick_ratio'], 3, labels=['Small Wick', 'Med Wick', 'Big Wick'], duplicates='drop')
        pivot2 = df.pivot_table(
            index='abs_class', 
            columns='wick_class', 
            values='y_reversion', 
            aggfunc='mean',
            observed=True
        ) * 100
        print("Reversion Rate (%):")
        print(pivot2.to_string(float_format=lambda x: f"{x:.2f}"))
    except Exception as e:
        print(f"Interaction failed: {e}")

if __name__ == "__main__":
    main()
