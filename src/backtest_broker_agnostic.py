"""
Backtest: Broker-Agnostic Edge Discovery
=========================================
Tests Price/Time enhancements to the Limit-at-Open (1:2 RR) strategy:
1. Macro Trend Filter (using 2X Macro Renko)
2. Dynamic Break-Even Trailing Stop
3. Trade Duration Tracking for Time-Based Invalidation

Outputs a detailed trade log for timeout analysis.
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

# Constants (Configurable via args)
MICRO_K = 0.00118
MACRO_K = 0.00236

# Toggles for this run
USE_MACRO_FILTER = True
USE_BE_TRAILING = True

class PendingOrder:
    def __init__(self, order_id, direction, limit_price, tp_price, sl_price,
                 brick_size, created_t_msc, macro_trend):
        self.order_id = order_id
        self.direction = direction        # +1 = buy, -1 = sell
        self.limit_price = limit_price    # Entry (brick open)
        self.tp_price = tp_price          # Take-profit (+2R)
        self.sl_price = sl_price          # Stop-loss (-1R)
        self.original_sl = sl_price
        self.brick_size = brick_size
        self.created_t_msc = created_t_msc
        self.macro_trend = macro_trend    # Trend at order creation
        
        # Calculate the +1R milestone price for BE trailing
        self.plus_1r_price = limit_price + brick_size if direction == 1 else limit_price - brick_size
        
        self.filled = False
        self.fill_t_msc = None
        self.hit_1r_t_msc = None
        self.exit_t_msc = None
        self.result = None
        self.pnl_R = 0.0

def run_backtest(tick_path: Path):
    logger.info(f"Loading tick data from {tick_path}...")
    df = pd.read_parquet(tick_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    logger.info(f"Loaded {len(df):,} ticks across {len(days)} trading days.")
    logger.info(f"Settings: MACRO_FILTER={USE_MACRO_FILTER}, BE_TRAILING={USE_BE_TRAILING}")

    daily_groups = {}
    for day, group in df.groupby('utc_day'):
        daily_groups[day] = group

    all_orders = []
    pending_orders = []
    active_trades = []
    order_counter = 0

    ORDER_EXPIRY_MS = 12 * 3600 * 1000
    TRADE_TIMEOUT_MS = 24 * 3600 * 1000  # Hard timeout

    t0 = time.time()

    for day_idx, day in enumerate(days):
        if day not in daily_groups:
            continue
        day_df = daily_groups[day]
        day_ticks = day_df.to_dict('records')
        if len(day_ticks) < 100:
            continue

        # Lookback for path optimization
        start_lb = day - timedelta(days=7)
        lb_dfs = []
        curr = start_lb
        while curr < day:
            if curr in daily_groups:
                lb_dfs.append(daily_groups[curr])
            curr += timedelta(days=1)

        if lb_dfs:
            lb_df = pd.concat(lb_dfs, ignore_index=True)
            lookback_ticks = lb_df.to_dict('records')
        else:
            lookback_ticks = []

        day_open = day_ticks[0]['bid']
        micro_brick_size = day_open * MICRO_K
        macro_brick_size = day_open * MACRO_K

        # Optimize anchor using micro brick size
        bridge.path_optimizer.K_MULTIPLIER = MICRO_K
        optimizer = PathOptimizer()
        if len(lookback_ticks) > 1000:
            best_price, _, _ = optimizer.find_optimal_anchor(lookback_ticks, micro_brick_size)
            if best_price is None:
                best_price = day_open
        else:
            best_price = day_open

        # Initialize both Renkos
        micro_renko = RenkoBuilder(best_price)
        micro_renko.update_brick_size(micro_brick_size, new_day_open=best_price)
        
        macro_renko = RenkoBuilder(best_price)
        macro_renko.update_brick_size(macro_brick_size, new_day_open=best_price)

        current_macro_trend = 0

        # Warmup
        for tick in lookback_ticks:
            micro_renko.update_tick(tick['bid'], tick['time_msc'])
            m_bricks = macro_renko.update_tick(tick['bid'], tick['time_msc'])
            for b in m_bricks:
                current_macro_trend = 1 if b.uptrend == 1 else -1

        # Day process
        for tick in day_ticks:
            bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']

            # Update Macro trend
            m_bricks = macro_renko.update_tick(bid, t_msc)
            for b in m_bricks:
                current_macro_trend = 1 if b.uptrend == 1 else -1

            # 1. Check pending orders
            still_pending = []
            for order in pending_orders:
                if (t_msc - order.created_t_msc) > ORDER_EXPIRY_MS:
                    order.result = 'expired'
                    all_orders.append(order)
                    continue

                filled = False
                if order.direction == 1:
                    if ask <= order.limit_price:
                        filled = True
                else:
                    if bid >= order.limit_price:
                        filled = True

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
                    trade.exit_t_msc = t_msc
                    trade.pnl_R = 0.0
                    all_orders.append(trade)
                    continue

                resolved = False
                if trade.direction == 1:  # Long
                    # Check +1R milestone for BE trailing
                    if not trade.hit_1r_t_msc and bid >= trade.plus_1r_price:
                        trade.hit_1r_t_msc = t_msc
                        if USE_BE_TRAILING:
                            trade.sl_price = trade.limit_price  # Move SL to breakeven

                    # Check TP / SL
                    if bid >= trade.tp_price:
                        trade.result = 'win'
                        trade.pnl_R = +2.0
                        trade.exit_t_msc = t_msc
                        resolved = True
                    elif bid <= trade.sl_price:
                        if trade.sl_price == trade.limit_price:
                            trade.result = 'breakeven'
                            trade.pnl_R = 0.0
                        else:
                            trade.result = 'loss'
                            trade.pnl_R = -1.0
                        trade.exit_t_msc = t_msc
                        resolved = True
                else:  # Short
                    # Check +1R milestone for BE trailing
                    if not trade.hit_1r_t_msc and ask <= trade.plus_1r_price:
                        trade.hit_1r_t_msc = t_msc
                        if USE_BE_TRAILING:
                            trade.sl_price = trade.limit_price

                    if ask <= trade.tp_price:
                        trade.result = 'win'
                        trade.pnl_R = +2.0
                        trade.exit_t_msc = t_msc
                        resolved = True
                    elif ask >= trade.sl_price:
                        if trade.sl_price == trade.limit_price:
                            trade.result = 'breakeven'
                            trade.pnl_R = 0.0
                        else:
                            trade.result = 'loss'
                            trade.pnl_R = -1.0
                        trade.exit_t_msc = t_msc
                        resolved = True

                if resolved:
                    all_orders.append(trade)
                else:
                    still_active.append(trade)
            active_trades = still_active

            # 3. Micro Bricks -> Place Orders
            new_micro = micro_renko.update_tick(bid, t_msc)
            for brick in new_micro:
                order_counter += 1
                direction = 1 if brick.uptrend == 1 else -1
                
                # Apply Macro Trend Filter if enabled
                if USE_MACRO_FILTER:
                    if direction != current_macro_trend:
                        continue  # Skip trade against macro trend

                if direction == 1:
                    limit_price = brick.open
                    tp_price = brick.open + 2 * micro_brick_size
                    sl_price = brick.open - 1 * micro_brick_size
                else:
                    limit_price = brick.open
                    tp_price = brick.open - 2 * micro_brick_size
                    sl_price = brick.open + 1 * micro_brick_size

                order = PendingOrder(
                    order_id=order_counter,
                    direction=direction,
                    limit_price=limit_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    brick_size=micro_brick_size,
                    created_t_msc=t_msc,
                    macro_trend=current_macro_trend
                )
                pending_orders.append(order)

        if (day_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info(f"  Day {day_idx+1}/{len(days)} ({day}) | "
                        f"Resolved: {len(all_orders)} | Active: {len(active_trades)} | {elapsed:.1f}s")

    for trade in active_trades:
        trade.result = 'timeout'
        trade.pnl_R = 0.0
        all_orders.append(trade)
    for order in pending_orders:
        order.result = 'expired'
        all_orders.append(order)

    elapsed = time.time() - t0
    logger.info(f"\nBacktest complete in {elapsed:.1f}s")

    # Save detailed trade log
    filled = [o for o in all_orders if o.filled]
    
    trade_data = []
    for o in filled:
        trade_data.append({
            'order_id': o.order_id,
            'direction': o.direction,
            'macro_trend': o.macro_trend,
            'fill_t_msc': o.fill_t_msc,
            'exit_t_msc': o.exit_t_msc,
            'duration_ms': o.exit_t_msc - o.fill_t_msc if o.exit_t_msc else 0,
            'hit_1r': o.hit_1r_t_msc is not None,
            'time_to_1r_ms': o.hit_1r_t_msc - o.fill_t_msc if o.hit_1r_t_msc else None,
            'result': o.result,
            'pnl_R': o.pnl_R
        })
        
    df_trades = pd.DataFrame(trade_data)
    out_path = BASE_DIR / "outputs" / f"trades_k{MICRO_K}_macro{int(USE_MACRO_FILTER)}_be{int(USE_BE_TRAILING)}.parquet"
    out_path.parent.mkdir(exist_ok=True)
    df_trades.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df_trades)} trades to {out_path}")

    # Metrics
    wins = [o for o in filled if o.result == 'win']
    losses = [o for o in filled if o.result == 'loss']
    be = [o for o in filled if o.result == 'breakeven']
    
    tradeable = len(wins) + len(losses) + len(be)
    if tradeable > 0:
        win_rate = len(wins) / tradeable * 100
        loss_rate = len(losses) / tradeable * 100
        be_rate = len(be) / tradeable * 100
        
        total_pnl_R = sum(o.pnl_R for o in filled)
        expectancy = total_pnl_R / tradeable
        
        logger.info(f"\n  ── Key Metrics ──")
        logger.info(f"  Total Trades:           {tradeable:,}")
        logger.info(f"  Win Rate:               {win_rate:.2f}% ({len(wins)} trades)")
        logger.info(f"  Loss Rate:              {loss_rate:.2f}% ({len(losses)} trades)")
        logger.info(f"  Break-Even Rate:        {be_rate:.2f}% ({len(be)} trades)")
        logger.info(f"  Per-Trade Expectancy:   {expectancy:+.4f} R")
        logger.info(f"  Total P&L:              {total_pnl_R:+.1f} R")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-macro", action="store_true")
    parser.add_argument("--no-be", action="store_true")
    parser.add_argument("--micro-k", type=float, default=0.00118)
    args = parser.parse_args()
    
    if args.no_macro: USE_MACRO_FILTER = False
    if args.no_be: USE_BE_TRAILING = False
    MICRO_K = args.micro_k
    MACRO_K = MICRO_K * 2
    
    tick_path = BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"
    run_backtest(tick_path)
