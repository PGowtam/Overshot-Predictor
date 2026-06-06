import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def evaluate_portfolio(lookback_days=50):
    t0 = time.time()
    
    parquet_path = BASE_DIR / "Data" / "blackbull_bricks_2026.parquet"
    logger.info(f"Loading bricks from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} bricks.")
    
    logger.info(f"Calculating {lookback_days}-Day calendar rolling percentiles...")
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
    
    logger.info(f"Found {len(signals)} raw executable signals. Simulating Portfolio with Trade Locking...")
    
    portfolio_log = []
    lock_release_time = 0
    
    for _, sig in signals.iterrows():
        entry_time = sig['timestamp']
        
        if entry_time < lock_release_time:
            continue
            
        direction = sig['direction']
        entry_price = sig['price']
        bs = sig['brick_size']
        utc_day = sig['utc_day']
        
        trade_dir = -1 if direction == 1 else 1
        sl = entry_price + bs if trade_dir == -1 else entry_price - bs
        
        # Custom Portfolio: 1x on 1R, 2x on 2R, 1x on 3R. Total Risk = 4 units.
        trades = {
            '1R': {'target': 1.0, 'vol': 1.0, 'tp': entry_price - (bs * 1.0) if trade_dir == -1 else entry_price + (bs * 1.0), 'status': 'OPEN', 'exit_time': None, 'pnl': 0},
            '2R': {'target': 2.0, 'vol': 2.0, 'tp': entry_price - (bs * 2.0) if trade_dir == -1 else entry_price + (bs * 2.0), 'status': 'OPEN', 'exit_time': None, 'pnl': 0},
            '3R': {'target': 3.0, 'vol': 1.0, 'tp': entry_price - (bs * 3.0) if trade_dir == -1 else entry_price + (bs * 3.0), 'status': 'OPEN', 'exit_time': None, 'pnl': 0}
        }
        
        future_df = df[df['timestamp'] > entry_time]
        open_count = 3
        max_exit_time = entry_time
        
        for _, future_brick in future_df.iterrows():
            f_price = future_brick['price']
            f_time = future_brick['timestamp']
            
            for key, trade in trades.items():
                if trade['status'] == 'OPEN':
                    if trade_dir == 1: 
                        if f_price <= sl: 
                            trade['status'] = 'LOSS'; trade['exit_time'] = f_time; trade['pnl'] = -1.0 * trade['vol']; open_count -= 1
                        elif f_price >= trade['tp']: 
                            trade['status'] = 'WIN'; trade['exit_time'] = f_time; trade['pnl'] = trade['target'] * trade['vol']; open_count -= 1
                    else: 
                        if f_price >= sl: 
                            trade['status'] = 'LOSS'; trade['exit_time'] = f_time; trade['pnl'] = -1.0 * trade['vol']; open_count -= 1
                        elif f_price <= trade['tp']: 
                            trade['status'] = 'WIN'; trade['exit_time'] = f_time; trade['pnl'] = trade['target'] * trade['vol']; open_count -= 1
                            
                    if trade['status'] != 'OPEN' and f_time > max_exit_time:
                        max_exit_time = f_time
                        
            if open_count == 0:
                break
                
        portfolio_pnl = sum(t['pnl'] for t in trades.values())
        portfolio_log.append({
            'utc_day': utc_day,
            'entry_time': pd.to_datetime(entry_time, unit='ms'),
            'exit_time': pd.to_datetime(max_exit_time, unit='ms'),
            'pnl_r': portfolio_pnl
        })
        
        lock_release_time = max_exit_time
        
    trades_df = pd.DataFrame(portfolio_log)
    
    if trades_df.empty:
        logger.info("No trades generated.")
        return
        
    total_sets = len(trades_df)
    total_pnl = trades_df['pnl_r'].sum()
    profitable_sets = sum(trades_df['pnl_r'] > 0)
    win_rate = (profitable_sets / total_sets) * 100
    
    gross_profit = trades_df[trades_df['pnl_r'] > 0]['pnl_r'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_r'] < 0]['pnl_r'].sum())
    pf = gross_profit / (gross_loss + 1e-8)
    
    logger.info(f"\n--- COMBINED PORTFOLIO RESULTS ({lookback_days}-Day Memory) ---")
    logger.info(f" Allocation: 1x on 1R | 2x on 2R | 1x on 3R (Total Risk: 4R)")
    logger.info(f" Total Signal Sets Traded: {total_sets}")
    logger.info(f" Portfolio Win Rate (Sets ending in net profit): {win_rate:.2f}%")
    logger.info(f" Portfolio Profit Factor: {pf:.2f}")
    logger.info(f" Net Portfolio PnL: {total_pnl:+.2f}R")
    
    logger.info(f"\nTotal script time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    evaluate_portfolio(30)
