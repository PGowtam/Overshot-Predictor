"""
Track A: Rule-Based Logic Tester
Evaluates manual rules over exhaustion scalars to see if a simple logic-based strategy
can achieve 70-75% hit rate with PF > 2.
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

def evaluate_rule(df, rule_mask, rule_name):
    total_samples = len(df)
    matched = df[rule_mask]
    trade_count = len(matched)
    
    if trade_count == 0:
        logger.info(f"[{rule_name}] -> 0 matches")
        return 0, 0, 0, 0
        
    coverage = (trade_count / total_samples) * 100
    wins = matched['y_reversion'].sum()
    losses = trade_count - wins
    hit_rate = (wins / trade_count) * 100
    pf = wins / (losses + 1e-8)
    
    logger.info(f"[{rule_name:<40}] Coverage: {coverage:>5.2f}% ({trade_count:>5} trades) | Hit Rate: {hit_rate:>5.2f}% | PF: {pf:.2f}")
    return coverage, hit_rate, pf, trade_count

def generate_frontier(df, feature, percentile_range, ascending=True):
    results = []
    for pt in percentile_range:
        if ascending:
            threshold = df[feature].quantile(pt / 100.0)
            mask = df[feature] <= threshold
        else:
            threshold = df[feature].quantile(1.0 - (pt / 100.0))
            mask = df[feature] >= threshold
            
        wins = df.loc[mask, 'y_reversion'].sum()
        trades = mask.sum()
        if trades > 0:
            hr = wins / trades
            pf = wins / (trades - wins + 1e-8)
            results.append((pt, trades, hr*100, pf))
            
    return results

def main():
    logger.info("Loading V3 labels for Rule Testing...")
    files = glob.glob(str(V3_LBL_DIR / "v3_labels_*.parquet"))
    if not files:
        logger.error("No V3 labels found!")
        return
        
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    df['y_reversion'] = df['path_class'].isin([1, 2]).astype(int)
    df = df.dropna(subset=['y_reversion', 'ofi_peak', 'vel_peak', 'spread_current', 'absorption_index', 'wick_ratio'])
    df['abs_ofi_peak'] = df['ofi_peak'].abs()
    
    # Calculate Quartiles
    spread_q75 = df['spread_current'].quantile(0.75)
    ofi_q25 = df['abs_ofi_peak'].quantile(0.25)
    vel_q75 = df['vel_peak'].quantile(0.75)
    wick_q75 = df['wick_ratio'].quantile(0.75)
    abs_q25 = df['absorption_index'].quantile(0.25)
    
    # Precompute masks
    high_spread = df['spread_current'] >= spread_q75
    low_ofi = df['abs_ofi_peak'] <= ofi_q25
    high_vel = df['vel_peak'] >= vel_q75
    big_wick = df['wick_ratio'] >= wick_q75
    low_abs = df['absorption_index'] <= abs_q25
    
    logger.info(f"Total Samples: {len(df)}")
    
    logger.info("\n--- Single Factor Rules ---")
    evaluate_rule(df, high_spread, "Spread > Q75")
    evaluate_rule(df, low_ofi, "OFI < Q25 (Low OFI)")
    evaluate_rule(df, high_vel, "Velocity > Q75")
    evaluate_rule(df, big_wick, "Wick Ratio > Q75")
    
    logger.info("\n--- Two-Factor Rules ---")
    evaluate_rule(df, high_spread & low_ofi, "High Spread & Low OFI")
    evaluate_rule(df, high_spread & big_wick, "High Spread & Big Wick")
    evaluate_rule(df, low_ofi & high_vel, "Low OFI & High Velocity")
    evaluate_rule(df, low_abs & big_wick, "Low Absorption & Big Wick")
    
    logger.info("\n--- Three-Factor Rules ---")
    evaluate_rule(df, high_spread & low_ofi & high_vel, "High Spread & Low OFI & High Velocity")
    evaluate_rule(df, high_spread & low_ofi & big_wick, "High Spread & Low OFI & Big Wick")
    evaluate_rule(df, high_spread & low_abs & big_wick, "High Spread & Low Abs & Big Wick")
    
    logger.info("\n--- RULE FRONTIER: Spread Expansion ---")
    frontier_range = list(range(1, 21))
    results = generate_frontier(df, 'spread_current', frontier_range, ascending=False)
    summary_tiers = [1, 2, 3, 5, 10, 15, 20]
    
    logger.info("\nTop % | Trades | Hit Rate | PF")
    logger.info("-" * 40)
    for res in results:
        pt, trades, hr, pf = res
        if pt in summary_tiers:
            logger.info(f"{pt:^5d} | {trades:^6d} | {hr:^7.2f}% | {pf:.2f}")

    logger.info("\n--- RULE FRONTIER: Low Proxy OFI ---")
    results_ofi = generate_frontier(df, 'abs_ofi_peak', frontier_range, ascending=True)
    logger.info("\nBot % | Trades | Hit Rate | PF")
    logger.info("-" * 40)
    for res in results_ofi:
        pt, trades, hr, pf = res
        if pt in summary_tiers:
            logger.info(f"{pt:^5d} | {trades:^6d} | {hr:^7.2f}% | {pf:.2f}")

if __name__ == "__main__":
    main()
