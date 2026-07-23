import os
import sys
import logging
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

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
sys.path.insert(0, str(BASE_DIR / "src"))
from regime_tracker_v4 import RegimeTrackerV4

logger = logging.getLogger(__name__)

class SimTrade_RR_Broker:
    def __init__(self):
        self.active_positions = []
        self.trade_log = []
        self.next_ticket = 1000
    
    def execute(self, rr_target, direction, entry, sl, tp, brick_size, timestamp):
        pos = {
            "ticket": self.next_ticket,
            "rr_target": rr_target,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "brick_size": brick_size,
            "entry_time": timestamp
        }
        self.active_positions.append(pos)
        self.next_ticket += 1
    
    def check_ticks(self, bid, ask, t_msc):
        remaining = []
        for pos in self.active_positions:
            resolved = False
            if pos["direction"] == 1:
                if bid <= pos["sl"]: 
                    self._close(pos, "LOSS", pos["sl"], t_msc)
                    resolved = True
                elif bid >= pos["tp"]: 
                    self._close(pos, "WIN", pos["tp"], t_msc)
                    resolved = True
            elif pos["direction"] == -1:
                if ask >= pos["sl"]: 
                    self._close(pos, "LOSS", pos["sl"], t_msc)
                    resolved = True
                elif ask <= pos["tp"]: 
                    self._close(pos, "WIN", pos["tp"], t_msc)
                    resolved = True
            
            if not resolved:
                remaining.append(pos)
                
        self.active_positions = remaining
        
    def force_close_all(self, bid, ask, t_msc):
        for pos in self.active_positions:
            close_price = bid if pos["direction"] == 1 else ask
            self._close(pos, "FORCE_CLOSE", close_price, t_msc)
        self.active_positions = []
        
    def _close(self, pos, outcome, close_price, exit_time):
        pnl = (close_price - pos["entry"]) * pos["direction"]
        pnl_r = pnl / pos["brick_size"]
        record = {
            "rr_target": pos["rr_target"],
            "direction": pos["direction"],
            "outcome": outcome,
            "pnl_r": pnl_r,
            "entry_time": pos["entry_time"],
            "exit_time": exit_time
        }
        self.trade_log.append(record)

global_daily_groups = {}

def simulate_day(day):
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
    brick_size = day_open * bridge.renko.K_MULTIPLIER
    
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
    
    warmup_records = []
    ofi_peak = 0.0
    for i, tick in enumerate(lookback_ticks):
        feat = feature_engine.compute_vector(tick['bid'], tick['ask'], 0.0, 0.0, tick['time_msc'])
        if feat is not None:
            if abs(feat[0]) > abs(ofi_peak): ofi_peak = feat[0]
            
        if i >= best_idx:
            new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
            for brick in new_bricks:
                feature_engine.on_new_brick(brick)
                
                t_day = pd.to_datetime(brick.timestamp, unit='ms', utc=True).date()
                if t_day >= (day - timedelta(days=5)) and t_day < day:
                    warmup_records.append({
                        'utc_day': t_day,
                        'spread_current': tick['ask'] - tick['bid'],
                        'abs_ofi_peak': abs(ofi_peak)
                    })
                ofi_peak = 0.0
                
    tracker = RegimeTrackerV4(lookback_days=5)
    if warmup_records:
        df_hist = pd.DataFrame(warmup_records)
        tracker.refresh(day, df_hist)
        
    if not tracker.is_ready(min_samples=100):
        return []
        
    broker = SimTrade_RR_Broker()
    ofi_peak = 0.0
    last_trade_time = -1
    
    for tick in day_ticks:
        bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']
        feat = feature_engine.compute_vector(bid, ask, 0.0, 0.0, t_msc)
        if feat is not None:
            if abs(feat[0]) > abs(ofi_peak): ofi_peak = feat[0]
            
        if broker.active_positions:
            broker.check_ticks(bid, ask, t_msc)
            
        new_bricks = renko.update_tick(bid, t_msc)
        for brick in new_bricks:
            feature_engine.on_new_brick(brick)
            
            spread_current = ask - bid
            abs_ofi = abs(ofi_peak)
            
            sp_pct = tracker.get_percentile('spread_current', spread_current)
            op_pct = tracker.get_percentile('abs_ofi_peak', abs_ofi)
            
            if sp_pct >= 99 and op_pct <= 3:
                if brick.timestamp != last_trade_time:
                    last_trade_time = brick.timestamp
                    
                    is_buy = (brick.uptrend == 1)
                    bs = brick.brick_size
                    
                    rr_targets = [1.0, 1.5, 2.0, 3.0]
                    for rr in rr_targets:
                        # Mean Reversion Logic: If brick went UP, we SHORT. If DOWN, we BUY.
                        price = bid if is_buy else ask
                        sl = price + bs if is_buy else price - bs
                        tp = price - (bs * rr) if is_buy else price + (bs * rr)
                        direction = -1 if is_buy else 1
                        
                        broker.execute(f"{rr}R", direction, price, sl, tp, bs, brick.timestamp)
                    
            ofi_peak = 0.0
            
    if broker.active_positions:
        broker.force_close_all(day_ticks[-1]['bid'], day_ticks[-1]['ask'], day_ticks[-1]['time_msc'])
        
    return broker.trade_log

def main():
    if sys.platform == "darwin" and mp.get_start_method(allow_none=True) != 'fork':
        mp.set_start_method('fork', force=True)
        
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parquet_path = str(BASE_DIR / "Data" / "xauusd_ticks_dukascopy_2026.parquet")
    logger.info(f"Loading tick data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    
    global global_daily_groups
    logger.info("Grouping data by day...")
    global_daily_groups = {day: group for day, group in df.groupby('utc_day')}
    
    logger.info("Starting RR Sweep Simulation across 2026...")
    all_trades = []
    with mp.Pool(10) as pool:
        for i, daily_trades in enumerate(pool.imap_unordered(simulate_day, days)):
            if daily_trades:
                all_trades.extend(daily_trades)
            if (i+1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(days)} days...")
                
    if not all_trades:
        logger.info("No trades generated.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    
    for rr in ["1.0R", "1.5R", "2.0R", "3.0R"]:
        strat_df = trades_df[trades_df['rr_target'] == rr].copy()
        if strat_df.empty: continue
        
        resolved = strat_df[strat_df['outcome'].isin(['WIN', 'LOSS'])]
        total = len(resolved)
        wins = sum(resolved['outcome'] == 'WIN')
        losses = total - wins
        wr = (wins / total * 100) if total > 0 else 0
        
        gross_profit = resolved[resolved['pnl_r'] > 0]['pnl_r'].sum()
        gross_loss = abs(resolved[resolved['pnl_r'] < 0]['pnl_r'].sum())
        pf = gross_profit / (gross_loss + 1e-8)
        
        total_pnl = strat_df['pnl_r'].sum()
        
        logger.info(f"\n--- {rr} TARGET ---")
        logger.info(f" Trades: {total}")
        logger.info(f" Win Rate: {wr:.2f}%")
        logger.info(f" Profit Factor: {pf:.2f}")
        logger.info(f" Total PnL: {total_pnl:+.2f}R")

if __name__ == "__main__":
    main()
