import os
import sys
import logging
import multiprocessing as mp
import pandas as pd
from pathlib import Path
from datetime import timedelta
import time

BASE_DIR = Path(__file__).resolve().parent.parent
TRADER_DIR = BASE_DIR / "BrickOfTicks_Trader"
sys.path.insert(0, str(TRADER_DIR))

import bridge.renko
import bridge.path_optimizer
bridge.renko.K_MULTIPLIER = 0.00118
bridge.path_optimizer.K_MULTIPLIER = 0.00118

from bridge.renko import RenkoBuilder
from bridge.feature_engine import LiveFeatureEngine
from bridge.path_optimizer import PathOptimizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

global_daily_groups = {}

def process_day(day):
    day_df = global_daily_groups.get(day)
    if day_df is None: return []
    day_ticks = day_df.to_dict('records')
    
    if len(day_ticks) < 100: return []
    
    start_day = day - timedelta(days=7)
    lookback_dfs = []
    curr = start_day
    while curr < day:
        if curr in global_daily_groups:
            lookback_dfs.append(global_daily_groups[curr])
        curr += timedelta(days=1)
        
    if lookback_dfs:
        lb_df = pd.concat(lookback_dfs, ignore_index=True)
        lookback_ticks = lb_df.to_dict('records')
    else:
        lookback_ticks = []
        
    day_open = day_ticks[0]['bid']
    brick_size = day_open * 0.00118
    
    optimizer = PathOptimizer()
    if len(lookback_ticks) > 1000:
        best_price, best_idx, _ = optimizer.find_optimal_anchor(lookback_ticks, brick_size)
        if best_price is None: best_price = day_open; best_idx = 0
    else:
        best_price = day_open; best_idx = 0
        
    renko = RenkoBuilder(best_price)
    renko.update_brick_size(brick_size, new_day_open=best_price)
    feature_engine = LiveFeatureEngine()
    feature_engine.update_brick_size(brick_size)
    
    ofi_peak = 0.0
    for i, tick in enumerate(lookback_ticks):
        feat = feature_engine.compute_vector(tick['bid'], tick['ask'], 0.0, 0.0, tick['time_msc'])
        if feat is not None:
            if abs(feat[0]) > abs(ofi_peak): ofi_peak = feat[0]
            
        if i >= best_idx:
            new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
            for brick in new_bricks:
                feature_engine.on_new_brick(brick)
                ofi_peak = 0.0
                
    extracted_bricks = []
    
    for tick in day_ticks:
        bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']
        feat = feature_engine.compute_vector(bid, ask, 0.0, 0.0, t_msc)
        if feat is not None:
            if abs(feat[0]) > abs(ofi_peak): ofi_peak = feat[0]
            
        new_bricks = renko.update_tick(bid, t_msc)
        for brick in new_bricks:
            feature_engine.on_new_brick(brick)
            
            spread_current = ask - bid
            abs_ofi = abs(ofi_peak)
            
            is_buy = (brick.uptrend == 1)
            extracted_bricks.append({
                'timestamp': brick.timestamp,
                'utc_day': day,
                'price': bid if is_buy else ask,
                'direction': 1 if is_buy else -1,
                'brick_size': brick.brick_size,
                'spread_current': spread_current,
                'abs_ofi_peak': abs_ofi
            })
            
            ofi_peak = 0.0
            
    return extracted_bricks

def main():
    if sys.platform == "darwin" and mp.get_start_method(allow_none=True) != 'fork':
        mp.set_start_method('fork', force=True)
        
    parquet_path = BASE_DIR / "Data" / "xauusd_ticks_5ers_2026.parquet"
    logger.info(f"Loading tick data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    
    days = sorted(df['utc_day'].unique())
    logger.info(f"Loaded {len(df)} ticks across {len(days)} trading days.")
    
    global global_daily_groups
    for day, group in df.groupby('utc_day'):
        global_daily_groups[day] = group
        
    logger.info("Extracting Renko Bricks in Parallel...")
    t0 = time.time()
    
    cores = max(1, mp.cpu_count() - 2)
    with mp.Pool(processes=cores) as pool:
        results = pool.map(process_day, days)
        
    all_bricks = [b for day_bricks in results for b in day_bricks]
    bricks_df = pd.DataFrame(all_bricks)
    bricks_df = bricks_df.sort_values('timestamp').reset_index(drop=True)
    
    out_path = BASE_DIR / "Data" / "5ers_bricks_2026.parquet"
    bricks_df.to_parquet(out_path, index=False)
    
    logger.info(f"Extraction complete in {time.time()-t0:.2f}s!")
    logger.info(f"Saved {len(bricks_df)} discrete Renko bricks to {out_path}")

if __name__ == "__main__":
    main()
