import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    t0 = time.time()
    
    parquet_path = BASE_DIR / "outputs" / "sim_labels_v4" / "v4_percentiles_labels.parquet"
    if not parquet_path.exists():
        logger.error(f"File not found: {parquet_path}")
        return
        
    logger.info(f"Loading Dukascopy bricks from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Dukascopy dataframe might have different column names or UTC format
    # Let's check the columns and timestamps
    if 'utc_day' not in df.columns:
        # Generate utc_day if missing
        if 'timestamp' in df.columns:
            # Check if timestamp is in ms or datetime
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                df['utc_day'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
            else:
                df['utc_day'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                
    logger.info(f"Loaded {len(df)} bricks. Date range: {df['utc_day'].min()} to {df['utc_day'].max()}")
    
    logger.info("Calculating 100-Day calendar rolling percentiles...")
    days = sorted(df['utc_day'].unique())
    signal_mask = pd.Series(False, index=df.index)
    
    # We might not want to run the full 2020-2026 for speed if it takes too long,
    # but let's try the whole thing.
    
    for i in range(100, len(days)):
        target_day = days[i]
        lookback_window = days[i-100:i]
        
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
        
        # 99/3 Rule for vacuum
        day_signals = (sp_pcts >= 99.0) & (op_pcts <= 3.0)
        signal_mask.loc[target_df.index] = day_signals

    signals = df[signal_mask].copy()
    
    # Deduplicate millisecond timestamps (the "gap fill" illusion fix)
    signals = signals.drop_duplicates(subset=['timestamp'])
    
    logger.info(f"Found {len(signals)} raw executable signals.")
    
    if len(signals) == 0:
        return
        
    # Extract the UTC hour from the timestamp
    if pd.api.types.is_numeric_dtype(signals['timestamp']):
        signals['dt'] = pd.to_datetime(signals['timestamp'], unit='ms')
    else:
        signals['dt'] = pd.to_datetime(signals['timestamp'])
        
    signals['hour'] = signals['dt'].dt.hour
    
    hour_counts = signals['hour'].value_counts().sort_index()
    
    logger.info("\n--- DUKASCOPY SIGNAL TIMESTAMPS (100-DAY MEMORY) ---")
    logger.info("Hour (UTC) | Trade Count")
    logger.info("-" * 25)
    for hr, count in hour_counts.items():
        logger.info(f"   {hr:02d}:00   | {count}")
        
    logger.info(f"\nTotal trades: {len(signals)}")
    logger.info(f"Total script time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
