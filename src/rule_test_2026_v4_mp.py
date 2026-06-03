"""
Phase 8: Live Trading Engine Upgrade (V4 Dynamic Regime)
========================================================
Runs the Pure V4 (Rule A) and V4 + Session Filter (Rule B) simultaneously
over the 2026 out-of-sample tick dataset using multiprocessing.
"""
import os
import sys
import logging
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
OUTPUT_DIR = BASE_DIR / "outputs" / "experiments" / "rule_test_2026_v4"

class SimTrade_V4_Broker:
    def __init__(self):
        self.active_positions = []
        self.trade_log = []
        self.next_ticket = 1000
    
    def execute(self, strategy, direction, price, sl, tp, brick_size, timestamp, scalars):
        pos = {
            "ticket": self.next_ticket,
            "strategy": strategy,
            "direction": direction,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "brick_size": brick_size,
            "entry_time": timestamp,
            "scalars": scalars 
        }
        self.active_positions.append(pos)
        self.next_ticket += 1
        return pos
    
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
            "ticket": pos["ticket"],
            "strategy": pos["strategy"],
            "direction": pos["direction"],
            "entry": pos["entry"],
            "sl": pos["sl"],
            "tp": pos["tp"],
            "brick_size": pos["brick_size"],
            "outcome": outcome,
            "close_price": close_price,
            "pnl_pts": pnl,
            "pnl_r": pnl_r,
            "entry_time": pos["entry_time"],
            "exit_time": exit_time
        }
        record.update(pos["scalars"])
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
                
                # NOTE: tick['ask'] - tick['bid'] exactly matches the training label 
                # logic since it evaluates precisely on the brick-closing tick.
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
        
    broker = SimTrade_V4_Broker()
    ofi_peak = 0.0
    weak_hours = {3, 15, 18, 22, 23}
    
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
            
            brick_dt = pd.to_datetime(brick.timestamp, unit='ms', utc=True)
            hour = brick_dt.hour
            weekday = brick_dt.day_name()
            
            if sp_pct >= 95 and op_pct <= 20:
                is_buy = (brick.uptrend == 1)
                bs = brick.brick_size
                price, sl, tp, direction = (bid, bid+bs, bid-bs, -1) if is_buy else (ask, ask-bs, ask+bs, 1)
                
                scalars = {
                    "spread_pct": sp_pct, 
                    "ofi_pct": op_pct,
                    "spread_raw": spread_current,
                    "abs_ofi_raw": abs_ofi,
                    "hour": hour,
                    "weekday": weekday,
                    "signal_time": brick.timestamp,
                    "brick_direction": brick.uptrend
                }
                
                broker.execute("Rule A", direction, price, sl, tp, bs, brick.timestamp, scalars)
                
                if hour not in weak_hours:
                    broker.execute("Rule B", direction, price, sl, tp, bs, brick.timestamp, scalars)
                    
            ofi_peak = 0.0
            
    if broker.active_positions:
        broker.force_close_all(day_ticks[-1]['bid'], day_ticks[-1]['ask'], day_ticks[-1]['time_msc'])
        
    return broker.trade_log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    
    if sys.platform == "darwin" and mp.get_start_method(allow_none=True) != 'fork':
        mp.set_start_method('fork', force=True)
        
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    parquet_path = str(BASE_DIR / "Data" / "xauusd_ticks_2026.parquet")
    logger.info(f"Loading tick data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    
    global global_daily_groups
    logger.info("Grouping data by day (optimizing memory transfer)...")
    global_daily_groups = {day: group for day, group in df.groupby('utc_day')}
    
    tasks = days
    logger.info(f"Prepared {len(tasks)} days for multiprocessing.")
    
    all_trades = []
    with mp.Pool(args.workers) as pool:
        for i, daily_trades in enumerate(pool.imap_unordered(simulate_day, tasks)):
            if daily_trades:
                all_trades.extend(daily_trades)
            if (i+1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(tasks)} days...")
                
    if not all_trades:
        logger.info("No trades generated.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    
    for strat in ["Rule A", "Rule B"]:
        strat_df = trades_df[trades_df['strategy'] == strat].copy()
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
        
        force_closed = sum(strat_df['outcome'] == 'FORCE_CLOSE')
        fc_pct = (force_closed / len(strat_df)) * 100 if len(strat_df) > 0 else 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f" 2026 DYNAMIC OUT-OF-SAMPLE TEST: {strat}")
        logger.info(f" Total Trades: {len(strat_df)} (Resolved: {total})")
        logger.info(f" Wins: {wins} | Losses: {losses}")
        logger.info(f" Win Rate: {wr:.2f}%")
        logger.info(f" Profit Factor: {pf:.2f}")
        logger.info(f" Force Closed Trades: {force_closed} ({fc_pct:.2f}%)")
        logger.info(f" Total PnL: {total_pnl:+.2f}R")
        logger.info(f"{'='*60}")
        
    trades_df.to_parquet(OUTPUT_DIR / "rule_test_2026_v4_trades.parquet")
    logger.info("Saved all trades to rule_test_2026_v4_trades.parquet")

if __name__ == "__main__":
    main()
