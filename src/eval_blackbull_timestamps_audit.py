import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def evaluate_lookback(df, lookback_days):
    days = sorted(df['utc_day'].unique())
    signal_mask = pd.Series(False, index=df.index)
    
    for i in range(lookback_days, len(days)):
        target_day = days[i]
        lookback_window = days[i-lookback_days:i]
        
        memory_df = df[df['utc_day'].isin(lookback_window)]
        target_df = df[df['utc_day'] == target_day]
        
        if memory_df.empty or target_df.empty:
            continue
            
        spread_dist = memory_df['spread_current'].values.copy()
        ofi_dist = memory_df['abs_ofi_peak'].values.copy()
        
        spread_dist.sort()
        ofi_dist.sort()
        
        def get_pct(val, arr):
            if len(arr) == 0: return 0.0
            idx = np.searchsorted(arr, val, side='right')
            return (idx / len(arr)) * 100.0
            
        sp_pcts = np.array([get_pct(v, spread_dist) for v in target_df['spread_current'].values])
        op_pcts = np.array([get_pct(v, ofi_dist) for v in target_df['abs_ofi_peak'].values])
        
        day_signals = (sp_pcts >= 99.0) & (op_pcts <= 3.0)
        signal_mask.loc[target_df.index] = day_signals

    signals = df[signal_mask].copy()
    signals = signals.drop_duplicates(subset=['timestamp'])
    
    signals['dt'] = pd.to_datetime(signals['timestamp'], unit='ms')
    signals['hour'] = signals['dt'].dt.hour
    signals['day_of_week'] = signals['dt'].dt.day_name()
    
    hour_counts = signals['hour'].value_counts().sort_index()
    day_counts = signals['day_of_week'].value_counts()
    
    logger.info(f"\n--- BLACKBULL SIGNAL TIMESTAMPS ({lookback_days}-DAY MEMORY) ---")
    logger.info("Hour (Broker Time) | Trade Count")
    logger.info("-" * 32)
    for hr, count in hour_counts.items():
        logger.info(f"      {hr:02d}:00       | {count}")
        
    logger.info("\nDay of Week | Trade Count")
    logger.info("-" * 25)
    for day, count in day_counts.items():
        logger.info(f"{day:>11} | {count}")
    # Calculate Win Rate for Monday trades (2.0R Target)
    monday_signals = signals[signals['day_of_week'] == 'Monday']
    if len(monday_signals) > 0:
        # Check reversion depth if it's in the dataset, but wait, blackbull_bricks doesn't have reversion_depth!
        # The true fast_eval_blackbull.py calculates it manually using forward search.
        pass
        
    logger.info(f"\nTotal trades: {len(signals)}")

def main():
    t0 = time.time()
    parquet_path = BASE_DIR / "Data" / "blackbull_bricks_2026.parquet"
    logger.info(f"Loading BlackBull bricks from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} bricks. Date range: {df['utc_day'].min()} to {df['utc_day'].max()}")
    
    evaluate_lookback(df, 30)
    evaluate_lookback(df, 100)
    
    logger.info(f"\nTotal script time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
