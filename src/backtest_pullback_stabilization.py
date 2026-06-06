"""
Backtest: Phase 5 Pullback Stabilization Filter
===============================================
The "Falling Knife" Dodger.
Instead of starting the delay timer when the brick closes, we wait for the 
price to actually touch the entry level (brick open).
The moment it touches, the stabilization timer begins.
If it survives X minutes without hitting the SL, we execute at market.
"""

import sys
import logging
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent
TRADER_DIR = BASE_DIR / "BrickOfTicks_Trader"
sys.path.insert(0, str(TRADER_DIR))

import bridge.renko
import bridge.path_optimizer

from bridge.renko import RenkoBuilder
from bridge.path_optimizer import PathOptimizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class PendingOrder:
    def __init__(self, order_id, direction, limit_price, tp_price, sl_price, brick_size, created_t_msc, delay_mins):
        self.order_id = order_id
        self.direction = direction
        self.original_limit = limit_price
        self.limit_price = limit_price
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.brick_size = brick_size
        self.created_t_msc = created_t_msc
        self.delay_mins = delay_mins
        
        self.state = 'WAITING_FOR_TOUCH'  # WAITING_FOR_TOUCH -> STABILIZING -> ACTIVE -> FILLED
        self.touch_t_msc = None
        self.activation_t_msc = None
        
        self.filled = False
        self.fill_t_msc = None
        self.exit_t_msc = None
        self.result = None
        self.pnl_R = 0.0

def run_backtest(tick_path: Path, micro_k: float, delay_list: list):
    logger.info(f"Loading tick data from {tick_path}...")
    df = pd.read_parquet(tick_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    logger.info(f"K_MULTIPLIER: {micro_k} | DELAYS TO TEST: {delay_list} mins")

    daily_groups = {day: group for day, group in df.groupby('utc_day')}

    all_orders = []
    pending_orders = []
    active_trades = []
    order_counter = 0

    ORDER_EXPIRY_MS = 12 * 3600 * 1000
    TRADE_TIMEOUT_MS = 24 * 3600 * 1000

    t0 = time.time()

    for day_idx, day in enumerate(days):
        if day not in daily_groups: continue
        day_ticks = daily_groups[day].to_dict('records')
        if len(day_ticks) < 100: continue

        start_lb = day - timedelta(days=7)
        lb_dfs = [daily_groups[curr] for curr in pd.date_range(start_lb, day-timedelta(days=1)).date if curr in daily_groups]
        lookback_ticks = pd.concat(lb_dfs).to_dict('records') if lb_dfs else []

        day_open = day_ticks[0]['bid']
        brick_size = day_open * micro_k

        bridge.path_optimizer.K_MULTIPLIER = micro_k
        optimizer = PathOptimizer()
        if len(lookback_ticks) > 1000:
            best_price, _, _ = optimizer.find_optimal_anchor(lookback_ticks, brick_size)
            if best_price is None: best_price = day_open
        else:
            best_price = day_open

        renko = RenkoBuilder(best_price)
        renko.update_brick_size(brick_size, new_day_open=best_price)

        for tick in lookback_ticks:
            renko.update_tick(tick['bid'], tick['time_msc'])

        for tick in day_ticks:
            bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']

            # 1. Process pending orders state machine
            still_pending = []
            for order in pending_orders:
                if (t_msc - order.created_t_msc) > ORDER_EXPIRY_MS:
                    order.result = 'expired'
                    all_orders.append(order)
                    continue

                if order.state == 'WAITING_FOR_TOUCH':
                    missed = False
                    if order.direction == 1 and bid >= order.tp_price: missed = True
                    elif order.direction == -1 and ask <= order.tp_price: missed = True
                    
                    if missed:
                        order.result = 'invalidated_missed_trade'
                        all_orders.append(order)
                        continue

                    touched = False
                    if order.direction == 1 and ask <= order.original_limit: touched = True
                    elif order.direction == -1 and bid >= order.original_limit: touched = True
                    
                    if touched:
                        order.touch_t_msc = t_msc
                        order.activation_t_msc = t_msc + (order.delay_mins * 60 * 1000)
                        order.state = 'STABILIZING'
                    
                    still_pending.append(order)

                elif order.state == 'STABILIZING':
                    invalidated = False
                    if order.direction == 1:
                        if bid <= order.sl_price or bid >= order.tp_price: invalidated = True
                    else:
                        if ask >= order.sl_price or ask <= order.tp_price: invalidated = True
                        
                    if invalidated:
                        order.result = 'invalidated_during_stabilization'
                        all_orders.append(order)
                        continue
                        
                    if t_msc >= order.activation_t_msc:
                        order.state = 'ACTIVE'
                        # Enter immediately if price is better or equal
                        if order.direction == 1 and ask <= order.original_limit:
                            order.limit_price = ask
                            order.filled = True
                            order.fill_t_msc = t_msc
                            active_trades.append(order)
                            continue
                        elif order.direction == -1 and bid >= order.original_limit:
                            order.limit_price = bid
                            order.filled = True
                            order.fill_t_msc = t_msc
                            active_trades.append(order)
                            continue
                    
                    still_pending.append(order)

                elif order.state == 'ACTIVE':
                    filled = False
                    if order.direction == 1 and ask <= order.limit_price: filled = True
                    elif order.direction == -1 and bid >= order.limit_price: filled = True

                    if filled:
                        order.filled = True
                        order.fill_t_msc = t_msc
                        active_trades.append(order)
                    else:
                        still_pending.append(order)
                        
            pending_orders = still_pending

            # 2. Check active trades
            still_active = []
            for trade in active_trades:
                if (t_msc - trade.fill_t_msc) > TRADE_TIMEOUT_MS:
                    trade.result = 'timeout'
                    all_orders.append(trade)
                    continue

                resolved = False
                if trade.direction == 1:
                    if bid >= trade.tp_price:
                        trade.result = 'win'
                        trade.pnl_R = (trade.tp_price - trade.limit_price) / brick_size
                        resolved = True
                    elif bid <= trade.sl_price:
                        trade.result = 'loss'
                        trade.pnl_R = (trade.sl_price - trade.limit_price) / brick_size
                        resolved = True
                else:
                    if ask <= trade.tp_price:
                        trade.result = 'win'
                        trade.pnl_R = (trade.limit_price - trade.tp_price) / brick_size
                        resolved = True
                    elif ask >= trade.sl_price:
                        trade.result = 'loss'
                        trade.pnl_R = (trade.limit_price - trade.sl_price) / brick_size
                        resolved = True

                if resolved:
                    trade.exit_t_msc = t_msc
                    all_orders.append(trade)
                else:
                    still_active.append(trade)
            active_trades = still_active

            # 3. New Bricks -> New Orders
            for brick in renko.update_tick(bid, t_msc):
                direction = 1 if brick.uptrend == 1 else -1
                limit = brick.open
                tp = limit + 2 * brick_size if direction == 1 else limit - 2 * brick_size
                sl = limit - 1 * brick_size if direction == 1 else limit + 1 * brick_size
                
                for delay in delay_list:
                    order_counter += 1
                    pending_orders.append(PendingOrder(order_counter, direction, limit, tp, sl, brick_size, t_msc, delay))

        if (day_idx + 1) % 10 == 0:
            logger.info(f"Day {day_idx+1}/{len(days)} | Resolved: {len(all_orders)} | {time.time()-t0:.1f}s")

    for t in active_trades: t.result = 'timeout'; all_orders.append(t)
    for o in pending_orders: o.result = 'expired'; all_orders.append(o)

    # Group by delay and Output
    for delay in delay_list:
        delay_orders = [o for o in all_orders if o.delay_mins == delay]
        filled = [o for o in delay_orders if o.filled]
        wins = [o for o in filled if o.result == 'win']
        losses = [o for o in filled if o.result == 'loss']
        invalidated = [o for o in delay_orders if o.result == 'invalidated_during_stabilization']
        
        tradeable = len(wins) + len(losses)
        if tradeable > 0:
            win_rate = len(wins) / tradeable * 100
            total_pnl = sum(o.pnl_R for o in filled)
            avg_win = sum(o.pnl_R for o in wins)/len(wins) if wins else 0
            avg_loss = sum(o.pnl_R for o in losses)/len(losses) if losses else 0
            
            logger.info(f"\n── {delay}-MIN PULLBACK STABILIZATION RESULTS ──")
            logger.info(f"Total Generated Signals: {len(delay_orders)}")
            logger.info(f"Dodged Falling Knives:   {len(invalidated)} ({len(invalidated)/len(delay_orders)*100:.1f}%)")
            logger.info(f"Trades Filled:           {tradeable}")
            logger.info(f"Win Rate:                {win_rate:.2f}%")
            logger.info(f"Expectancy:              {total_pnl/tradeable:+.4f} R")
            logger.info(f"Total P&L:               {total_pnl:+.1f} R")
            logger.info(f"Avg Win:                 {avg_win:+.2f} R")
            logger.info(f"Avg Loss:                {avg_loss:+.2f} R")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delays", type=str, default="15,25,30,45,50")
    parser.add_argument("--micro-k", type=float, default=0.00118)
    args = parser.parse_args()
    
    delay_list = [int(x.strip()) for x in args.delays.split(",")]
    tick_path = BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"
    run_backtest(tick_path, args.micro_k, delay_list)
