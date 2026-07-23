import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging

BASE_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def evaluate_hardcoded_logic():
    t0 = time.time()
    
    parquet_path = BASE_DIR / "Data" / "5ers_bricks_2026.parquet"
    if not parquet_path.exists():
        logger.error(f"Waiting for {parquet_path} to be generated...")
        return
        
    logger.info(f"Loading 5ers bricks from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    logger.info(f"Loaded {len(df)} bricks. Date range: {df['utc_day'].min()} to {df['utc_day'].max()}")
    
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['day_name'] = df['dt'].dt.day_name()
    
    logger.info(f"Unique dates in dataset: {df['utc_day'].unique()}")
    logger.info(f"Unique days of week: {df['day_name'].unique()}")
    
    # Debug Monday 1 AM bricks
    monday_df = df[df['day_name'] == 'Monday']
    logger.info("\n--- MONDAY BRICKS ---")
    for idx, row in monday_df.iterrows():
        logger.info(f"{row['dt']} | Spread: {row['spread_current']:.2f} | OFI Peak: {row['abs_ofi_peak']:.2f}")
        
    mask = (
        (df['day_name'] == 'Monday') & 
        (df['hour'] == 1) & 
        (df['minute'] <= 5) & 
        (df['spread_current'] >= 1.5) & 
        (df['abs_ofi_peak'] <= 1.0)
    )
    
    signals = df[mask].copy()
    signals = signals.drop_duplicates(subset=['timestamp'])
    
    logger.info(f"Found {len(signals)} hardcoded signals.")
    
    if len(signals) == 0:
        logger.info("No trades found matching the hardcoded logic on the 5ers dataset.")
        return
        
    # Let's print the actual trades
    for idx, row in signals.iterrows():
        logger.info(f"TRADE: {row['dt']} | Spread: {row['spread_current']:.2f} | OFI: {row['abs_ofi_peak']:.2f} | Price: {row['close']}")
        
    # Simulate RR Targets
    logger.info("\nSimulating RR Targets by forward searching the raw bricks...")
    
    results = []
    
    for idx, trade in signals.iterrows():
        entry_time = trade['timestamp']
        entry_price = trade['close']
        trade_dir = 1 if trade['direction'] < 0 else -1 # Mean Reversion!
        
        # Forward search for this specific day
        future_df = df[(df['utc_day'] == trade['utc_day']) & (df['timestamp'] > entry_time)]
        
        hit_1r = False
        hit_2r = False
        hit_3r = False
        sl_hit = False
        
        target_1r = entry_price + (trade_dir * 1.0)
        target_2r = entry_price + (trade_dir * 2.0)
        target_3r = entry_price + (trade_dir * 3.0)
        sl_price = entry_price - (trade_dir * 1.0)
        
        for _, f_row in future_df.iterrows():
            if trade_dir == 1:
                # BUY Trade
                if f_row['low'] <= sl_price:
                    sl_hit = True
                    break
                if f_row['high'] >= target_3r:
                    hit_1r = True
                    hit_2r = True
                    hit_3r = True
                    break
                elif f_row['high'] >= target_2r:
                    hit_1r = True
                    hit_2r = True
                elif f_row['high'] >= target_1r:
                    hit_1r = True
            else:
                # SELL Trade
                if f_row['high'] >= sl_price:
                    sl_hit = True
                    break
                if f_row['low'] <= target_3r:
                    hit_1r = True
                    hit_2r = True
                    hit_3r = True
                    break
                elif f_row['low'] <= target_2r:
                    hit_1r = True
                    hit_2r = True
                elif f_row['low'] <= target_1r:
                    hit_1r = True
                    
        results.append({
            'dt': trade['dt'],
            '1r': hit_1r,
            '2r': hit_2r,
            '3r': hit_3r,
            'sl': sl_hit
        })
        
    res_df = pd.DataFrame(results)
    
    for rr_name, rr_val in [('1.0R', '1r'), ('2.0R', '2r'), ('3.0R', '3r')]:
        wins = res_df[rr_val].sum()
        total = len(res_df)
        losses = res_df['sl'].sum()
        
        wr = (wins / total) * 100 if total > 0 else 0
        pf = (wins * float(rr_name[:3])) / max(1, losses)
        pnl = (wins * float(rr_name[:3])) - losses
        
        logger.info(f"\n--- {rr_name} TARGET ---")
        logger.info(f" Trades: {total} | WR: {wr:.2f}% | PF: {pf:.2f} | PnL: {pnl:+.2f}R")

if __name__ == "__main__":
    evaluate_hardcoded_logic()
