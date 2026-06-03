"""
Phase 2 & 3: Dynamic Percentile Generation & Backtest (V4)
==========================================================
Loads the V3 scalar dataset, computes 5-day rolling percentiles using RegimeTrackerV4,
and compares the absolute baseline rule against the new Dynamic Regime Rule.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.regime_tracker_v4 import RegimeTrackerV4

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IN_DIR = BASE_DIR / "outputs" / "sim_labels_v3"
OUT_DIR = BASE_DIR / "outputs" / "sim_labels_v4"

def generate_percentiles():
    logger.info("Loading V3 labels for dynamic percentile computation...")
    files = glob.glob(str(IN_DIR / "v3_labels_*.parquet"))
    if not files:
        logger.error("No V3 labels found!")
        return None
        
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.dropna(subset=['path_class']).copy()
    df['reversion'] = df['path_class'].isin([1, 2]).astype(int)
    df['abs_ofi_peak'] = df['ofi_peak'].abs()
    
    if 'utc_day' not in df.columns:
        df['utc_day'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.date
        
    df = df.sort_values('timestamp').reset_index(drop=True)
    days = sorted(df['utc_day'].unique())
    
    tracker = RegimeTrackerV4(lookback_days=5)
    
    base_features = ['spread_current', 'abs_ofi_peak', 'vel_peak', 'wick_ratio', 'absorption_index']
    pct_cols = [f"{f}_pct" for f in base_features]
    for c in pct_cols:
        df[c] = np.nan
        
    logger.info(f"Computing rolling percentiles across {len(days)} trading days...")
    
    valid_days = 0
    for day in days:
        tracker.refresh(day, df)
        if not tracker.is_ready(min_samples=100): 
            continue
            
        day_mask = df['utc_day'] == day
        if not day_mask.any(): continue
        
        valid_days += 1
        for f in base_features:
            vals = df.loc[day_mask, f].values
            pcts = np.searchsorted(tracker.histories[f], vals, side='right') / len(tracker.histories[f]) * 100.0
            df.loc[day_mask, f"{f}_pct"] = pcts
            
    logger.info(f"Percentiles computed for {valid_days} active trading days.")
    df_clean = df.dropna(subset=pct_cols).copy()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "v4_percentiles_labels.parquet"
    df_clean.to_parquet(out_path)
    logger.info(f"Saved {len(df_clean)} samples to {out_path}")
    return df_clean

def print_metrics(df, mask, name):
    subset = df[mask]
    total = len(subset)
    if total == 0:
        logger.info(f"{name:.<40} 0 trades")
        return
        
    rev_rate = subset['reversion'].mean() * 100
    wins = subset['reversion'].sum()
    losses = total - wins
    pf = wins / max(losses, 1)
    
    logger.info(f"{name:.<40} {total:>5} trades | HR: {rev_rate:>5.2f}% | PF: {pf:>4.2f}")

def main():
    out_path = OUT_DIR / "v4_percentiles_labels.parquet"
    if out_path.exists():
        logger.info("Loading existing V4 percentile dataset...")
        df = pd.read_parquet(out_path)
    else:
        df = generate_percentiles()
        
    if df is None: return
    
    # Winsorize spread_current for the absolute rule benchmark (matches rule_stability)
    p99 = df['spread_current'].quantile(0.99)
    df['spread_current_w'] = df['spread_current'].clip(upper=p99)
    
    logger.info("\n" + "="*60)
    logger.info(" PHASE 3: DYNAMIC REGIME BACKTEST COMPARISON")
    logger.info("="*60)
    
    # 1. Absolute Baseline (from stat study)
    logger.info("\n[1] Absolute Threshold Baseline")
    mask_abs = (df['spread_current_w'] >= 0.784) & (df['abs_ofi_peak'] <= 1.074)
    print_metrics(df, mask_abs, "Spread >= 0.784 & OFI <= 1.074")
    
    # 2. Dynamic Equivalents
    logger.info("\n[2] Dynamic Regime Percentiles")
    mask_dyn1 = (df['spread_current_pct'] >= 75) & (df['abs_ofi_peak_pct'] <= 25)
    print_metrics(df, mask_dyn1, "Spread_pct >= 75 & OFI_pct <= 25")
    
    mask_dyn2 = (df['spread_current_pct'] >= 90) & (df['abs_ofi_peak_pct'] <= 10)
    print_metrics(df, mask_dyn2, "Spread_pct >= 90 & OFI_pct <= 10")
    
    mask_dyn3 = (df['spread_current_pct'] >= 95) & (df['abs_ofi_peak_pct'] <= 25)
    print_metrics(df, mask_dyn3, "Spread_pct >= 95 & OFI_pct <= 25")
    
    mask_dyn4 = (df['spread_current_pct'] >= 95) & (df['abs_ofi_peak_pct'] <= 25) & (df['vel_peak_pct'] <= 25)
    print_metrics(df, mask_dyn4, "Spread>=95 & OFI<=25 & Vel<=25")

if __name__ == "__main__":
    main()
