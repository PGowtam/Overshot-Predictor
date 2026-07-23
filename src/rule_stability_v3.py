"""
Track A4: Rule Stability
Evaluates the rules year-by-year to ensure PF and Hit Rate don't wildly fluctuate.
Also winsorizes spread at 99th percentile for the frontier sanity check.
"""
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
V3_LBL_DIR = BASE_DIR / "outputs" / "sim_labels_v3"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def eval_rule_yearly(df, rule_mask, rule_name):
    years = df['year'].unique()
    years = sorted(years)
    
    logger.info(f"\n--- {rule_name} ---")
    logger.info("Year | Trades | Hit Rate | PF")
    logger.info("-" * 35)
    
    for y in years:
        y_mask = df['year'] == y
        matched = df[rule_mask & y_mask]
        trades = len(matched)
        if trades == 0:
            logger.info(f"{y} |      0 |   0.00% | 0.00")
            continue
            
        wins = matched['y_reversion'].sum()
        losses = trades - wins
        hr = (wins / trades) * 100
        pf = wins / (losses + 1e-8)
        logger.info(f"{y} | {trades:^6d} | {hr:>6.2f}% | {pf:.2f}")

def main():
    logger.info("Loading V3 labels for Stability Testing...")
    files = glob.glob(str(V3_LBL_DIR / "v3_labels_*.parquet"))
    if not files:
        logger.error("No V3 labels found!")
        return
        
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    df['y_reversion'] = df['path_class'].isin([1, 2]).astype(int)
    df = df.dropna(subset=['y_reversion', 'ofi_peak', 'vel_peak', 'spread_current', 'absorption_index', 'wick_ratio', 'year'])
    df['abs_ofi_peak'] = df['ofi_peak'].abs()
    
    # Winsorize Spread at 99th percentile to remove extreme weekend/news gaps
    p99 = df['spread_current'].quantile(0.99)
    logger.info(f"Winsorizing spread_current at 99th percentile: {p99:.4f}")
    df['spread_winsorized'] = df['spread_current'].clip(upper=p99)
    
    # Re-calc quartiles with winsorized spread
    spread_q75 = df['spread_winsorized'].quantile(0.75)
    ofi_q25 = df['abs_ofi_peak'].quantile(0.25)
    vel_q75 = df['vel_peak'].quantile(0.75)
    
    high_spread = df['spread_winsorized'] >= spread_q75
    low_ofi = df['abs_ofi_peak'] <= ofi_q25
    high_vel = df['vel_peak'] >= vel_q75
    
    eval_rule_yearly(df, high_spread, "High Spread (> Q75)")
    eval_rule_yearly(df, high_spread & low_ofi, "High Spread & Low OFI")
    eval_rule_yearly(df, high_spread & low_ofi & high_vel, "High Spread & Low OFI & High Velocity")
    
    logger.info("\n--- WINSORIZED RULE FRONTIER: Spread Expansion ---")
    results = []
    frontier_range = list(range(1, 21))
    summary_tiers = [1, 2, 3, 5, 10, 15, 20]
    
    for pt in frontier_range:
        threshold = df['spread_winsorized'].quantile(1.0 - (pt / 100.0))
        mask = df['spread_winsorized'] >= threshold
        wins = df.loc[mask, 'y_reversion'].sum()
        trades = mask.sum()
        if trades > 0:
            hr = wins / trades
            pf = wins / (trades - wins + 1e-8)
            results.append((pt, trades, hr*100, pf))
            
    logger.info("Top % | Trades | Hit Rate | PF")
    logger.info("-" * 40)
    for res in results:
        pt, trades, hr, pf = res
        if pt in summary_tiers:
            logger.info(f"{pt:^5d} | {trades:^6d} | {hr:^7.2f}% | {pf:.2f}")

if __name__ == "__main__":
    main()
