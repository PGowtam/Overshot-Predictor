"""
Backtest: Phase 4 Time-Delayed Entry
====================================
Implements the 6-minute and 15-minute wait logic before placing a limit order.
If the price has hit the target or stop loss during the wait, the setup is invalidated.
If the price is at a BETTER level after the wait, we enter at market.
Otherwise, we place the limit order at the original open price.
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

MICRO_K = 0.00118
DELAY_MINS = 6

class PendingOrder:
    def __init__(self, order_id, direction, original_limit, tp_price, sl_price,
                 brick_size, created_t_msc):
        self.order_id = order_id
        self.direction = direction
        self.original_limit = original_limit
        self.limit_price = original_limit
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.brick_size = brick_size
        self.created_t_msc = created_t_msc
        
        self.activation_t_msc = created_t_msc + (DELAY_MINS * 60 * 1000)
        self.state = 'WAITING'  # WAITING -> ACTIVE -> FILLED
        
        self.filled = False
        self.fill_t_msc = None
        self.exit_t_msc = None
        self.result = None
        self.pnl_R = 0.0

def run_backtest(tick_path: Path):
    logger.info(f"Loading tick data from {tick_path}...")
    df = pd.read_parquet(tick_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    logger.info(f"Loaded {len(df):,} ticks. Delay: {DELAY_MINS} minutes.")

    daily_groups = {}
    for day, group in df.groupby('utc_day'):
        daily_groups[day] = group

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
        brick_size = day_open * MICRO_K

        bridge.path_optimizer.K_MULTIPLIER = MICRO_K
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

            # 1. Check pending orders
            still_pending = []
            for order in pending_orders:
                if (t_msc - order.created_t_msc) > ORDER_EXPIRY_MS:
                    order.result = 'expired'
                    all_orders.append(order)
                    continue

                if order.state == 'WAITING':
                    # Check invalidation during wait
                    invalidated = False
                    if order.direction == 1:
                        if bid >= order.tp_price or bid <= order.sl_price: invalidated = True
                    else:
                        if ask <= order.tp_price or ask >= order.sl_price: invalidated = True
                    
                    if invalidated:
                        order.result = 'invalidated'
                        all_orders.append(order)
                        continue
                    
                    # Activate if time reached
                    if t_msc >= order.activation_t_msc:
                        order.state = 'ACTIVE'
                        # Market execution check
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
                    # Normal limit fill check
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
                order_counter += 1
                direction = 1 if brick.uptrend == 1 else -1
                limit = brick.open
                tp = limit + 2 * brick_size if direction == 1 else limit - 2 * brick_size
                sl = limit - 1 * brick_size if direction == 1 else limit + 1 * brick_size
                
                pending_orders.append(PendingOrder(order_counter, direction, limit, tp, sl, brick_size, t_msc))

        if (day_idx + 1) % 10 == 0:
            logger.info(f"Day {day_idx+1}/{len(days)} | Resolved: {len(all_orders)} | {time.time()-t0:.1f}s")

    for t in active_trades: t.result = 'timeout'; all_orders.append(t)
    for o in pending_orders: o.result = 'expired'; all_orders.append(o)

    # Output
    filled = [o for o in all_orders if o.filled]
    wins = [o for o in filled if o.result == 'win']
    losses = [o for o in filled if o.result == 'loss']
    invalidated = [o for o in all_orders if o.result == 'invalidated']
    
    tradeable = len(wins) + len(losses)
    if tradeable > 0:
        win_rate = len(wins) / tradeable * 100
        total_pnl = sum(o.pnl_R for o in filled)
        
        logger.info(f"\n── {DELAY_MINS}-MIN DELAY RESULTS ──")
        logger.info(f"Total Generated Signals: {len(all_orders)}")
        logger.info(f"Invalidated during wait: {len(invalidated)} ({len(invalidated)/len(all_orders)*100:.1f}%)")
        logger.info(f"Trades Filled:           {tradeable}")
        logger.info(f"Win Rate:                {win_rate:.2f}%")
        logger.info(f"Expectancy:              {total_pnl/tradeable:+.4f} R")
        logger.info(f"Total P&L:               {total_pnl:+.1f} R")
        logger.info(f"Avg Win:                 {sum(o.pnl_R for o in wins)/len(wins):+.2f} R")
        logger.info(f"Avg Loss:                {sum(o.pnl_R for o in losses)/len(losses):+.2f} R")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=int, default=6)
    args = parser.parse_args()
    DELAY_MINS = args.delay
    tick_path = BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"
    run_backtest(tick_path)
