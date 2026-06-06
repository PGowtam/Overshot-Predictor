import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging

BASE_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def evaluate_baseline():
    t0 = time.time()
    
    parquet_path = BASE_DIR / "Data" / "5ers_bricks_2026.parquet"
    if not parquet_path.exists():
        logger.error(f"File not found: {parquet_path}")
        return
        
    logger.info(f"Loading 5ers bricks from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} bricks. Simulating 'Blind Reversal' with Trade Locking...")
    
    results = []
    
    active_trade = None
    
    for idx, row in df.iterrows():
        # 1. Resolve active trade if there is one
        if active_trade is not None:
            price = row['price']
            t_dir = active_trade['direction']
            sl = active_trade['sl']
            tp = active_trade['tp']
            
            resolved = False
            pnl = 0
            
            if t_dir == 1: # Long
                if price <= sl:
                    pnl = -1.0
                    resolved = True
                elif price >= tp:
                    pnl = 3.0
                    resolved = True
            else: # Short
                if price >= sl:
                    pnl = -1.0
                    resolved = True
                elif price <= tp:
                    pnl = 3.0
                    resolved = True
                    
            if resolved:
                active_trade['pnl'] = pnl
                active_trade['exit_time'] = row['timestamp']
                results.append(active_trade)
                active_trade = None
        
        # 2. If no active trade, take a new blind reversal trade on the CURRENT brick
        # The user wants to take a reversal trade for EVERY new brick generated.
        if active_trade is None:
            # Reversal logic: If brick went UP (direction 1), we go SHORT (-1)
            # If brick went DOWN (direction -1), we go LONG (1)
            t_dir = -1 if row['direction'] > 0 else 1
            
            entry_price = row['price']
            
            # Risk/Reward is 1:3. 1R is equal to the brick size!
            brick_size = row['brick_size']
            sl_price = entry_price - (t_dir * 1.0 * brick_size)
            tp_price = entry_price + (t_dir * 3.0 * brick_size)
            
            active_trade = {
                'entry_time': row['timestamp'],
                'entry_price': entry_price,
                'direction': t_dir,
                'sl': sl_price,
                'tp': tp_price
            }

    # If the final trade never resolved, we just drop it or mark it as unresolved
    
    if len(results) == 0:
        logger.info("No trades resolved.")
        return
        
    res_df = pd.DataFrame(results)
    wins = len(res_df[res_df['pnl'] == 3.0])
    losses = len(res_df[res_df['pnl'] == -1.0])
    total = wins + losses
    
    wr = (wins / total) * 100 if total > 0 else 0
    pnl = res_df['pnl'].sum()
    pf = (wins * 3.0) / max(1, losses)
    
    logger.info("\n--- BASELINE BLIND REVERSAL (5ERS) ---")
    logger.info(" Rules: Trade every brick | 1:3 RR | Inverse Direction | Trade Locking")
    logger.info(f" Total Trades Taken: {total}")
    logger.info(f" Wins (3R): {wins}")
    logger.info(f" Losses (1R): {losses}")
    logger.info(f" Win Rate: {wr:.2f}%")
    logger.info(f" Profit Factor: {pf:.2f}")
    logger.info(f" Net PnL: {pnl:+.2f}R")
    
    logger.info(f"\nTotal script time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    evaluate_baseline()
