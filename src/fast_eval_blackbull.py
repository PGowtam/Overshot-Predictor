import pandas as pd
import numpy as np
import time
from pathlib import Path
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def evaluate_fast():
    parquet_path = BASE_DIR / "Data" / "blackbull_bricks_2026.parquet"
    if not parquet_path.exists():
        logger.error(f"File not found: {parquet_path}. Run extract_bricks_blackbull.py first!")
        return
        
    logger.info(f"Loading bricks from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} bricks.")
    
    t0 = time.time()
    
    # 30-Day Rolling Percentiles (Calendar Based)
    logger.info("Calculating 30-Day calendar rolling percentiles...")
    
    days = sorted(df['utc_day'].unique())
    signal_mask = pd.Series(False, index=df.index)
    
    for i in range(30, len(days)):
        target_day = days[i]
        lookback_days = days[i-30:i]
        
        memory_df = df[df['utc_day'].isin(lookback_days)]
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
    
    # FILTER FOR MONDAY
    signals['dt'] = pd.to_datetime(signals['timestamp'], unit='ms')
    signals = signals[signals['dt'].dt.day_name() == 'Monday']
    
    logger.info(f"Found {len(signals)} raw executable signals on MONDAYS. Simulating RR...\n")
    
    trade_log = []
    
    for _, sig in signals.iterrows():
        entry_time = sig['timestamp']
        direction = sig['direction']
        entry_price = sig['price']
        bs = sig['brick_size']
        utc_day = sig['utc_day']
        
        future_df = df[df['timestamp'] > entry_time]
        
        for rr in [1.0, 2.0, 3.0]:
            trade_dir = -1 if direction == 1 else 1
            sl = entry_price + bs if trade_dir == -1 else entry_price - bs
            tp = entry_price - (bs * rr) if trade_dir == -1 else entry_price + (bs * rr)
            
            outcome = "OPEN"
            for _, future_brick in future_df.iterrows():
                f_price = future_brick['price']
                if trade_dir == 1: 
                    if f_price <= sl: outcome = "LOSS"; exit_time = future_brick['timestamp']; break
                    if f_price >= tp: outcome = "WIN"; exit_time = future_brick['timestamp']; break
                else: 
                    if f_price >= sl: outcome = "LOSS"; exit_time = future_brick['timestamp']; break
                    if f_price <= tp: outcome = "WIN"; exit_time = future_brick['timestamp']; break
                    
            if outcome != "OPEN":
                trade_log.append({
                    'utc_day': utc_day,
                    'rr_target': f"{rr}R", 
                    'outcome': outcome, 
                    'pnl_r': rr if outcome == "WIN" else -1.0,
                    'entry_time': pd.to_datetime(entry_time, unit='ms'),
                    'exit_time': pd.to_datetime(exit_time, unit='ms') if exit_time else None,
                    'entry_price': entry_price,
                    'direction': "BUY" if trade_dir == 1 else "SELL"
                })
                
    trades_df = pd.DataFrame(trade_log)
    
    # Save the 1.0R trades to a CSV
    if not trades_df.empty:
        r1_trades = trades_df[trades_df['rr_target'] == '1.0R']
        csv_path = BASE_DIR / "outputs" / "blackbull_100day_trades.csv"
        csv_path.parent.mkdir(exist_ok=True)
        r1_trades.to_csv(csv_path, index=False)
        logger.info(f"\nSaved detailed 1.0R trade log to {csv_path}")
        
    # Print daily breakdown for the 1.0R target to avoid duplicate counting
    logger.info("\n--- DAILY TRADE BREAKDOWN (Last 5 Days) ---")
    if not trades_df.empty:
        daily_counts = trades_df[trades_df['rr_target'] == '1.0R'].groupby('utc_day').size()
        for day, count in daily_counts.items():
            logger.info(f"  {day}: {count} trades")
    else:
        logger.info("  No trades found.")
        
    for rr in ["1.0R", "2.0R", "3.0R"]:
        strat_df = trades_df[trades_df['rr_target'] == rr]
        if strat_df.empty: continue
        total = len(strat_df)
        wins = sum(strat_df['outcome'] == 'WIN')
        wr = (wins / total * 100) if total > 0 else 0
        pf = strat_df[strat_df['pnl_r'] > 0]['pnl_r'].sum() / (abs(strat_df[strat_df['pnl_r'] < 0]['pnl_r'].sum()) + 1e-8)
        logger.info(f"\n--- {rr} TARGET ---")
        logger.info(f" Trades: {total} | WR: {wr:.2f}% | PF: {pf:.2f} | PnL: {strat_df['pnl_r'].sum():+.2f}R")

    logger.info(f"\nTotal script time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    evaluate_fast()
