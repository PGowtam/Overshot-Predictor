"""
Backtest: Trade the Fade (Mean Reversion) Strategy
===================================================
After every Renko brick closes, place a LIMIT order at the brick's OPEN price
in the OPPOSITE direction as the brick, targeting mean reversion (-1 brick size
from the open = the previous brick's open) with a stop at +1 brick size from
the open.

Risk:Reward = 1:1 (Risk 1 to make 1)

This tests whether the ~69% mathematical win rate of fading the move
can overcome the negative R:R and spread drag.
"""

import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent
TRADER_DIR = BASE_DIR / "BrickOfTicks_Trader"
sys.path.insert(0, str(TRADER_DIR))

import bridge.renko
import bridge.path_optimizer
bridge.renko.K_MULTIPLIER = 0.00118
bridge.path_optimizer.K_MULTIPLIER = 0.00118

from bridge.renko import RenkoBuilder, K_MULTIPLIER
from bridge.path_optimizer import PathOptimizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class PendingOrder:
    def __init__(self, order_id, direction, limit_price, tp_price, sl_price,
                 brick_size, created_t_msc):
        self.order_id = order_id
        self.direction = direction        # +1 = buy, -1 = sell
        self.limit_price = limit_price    # Entry (brick open)
        self.tp_price = tp_price          # Take-profit (reversal target)
        self.sl_price = sl_price          # Stop-loss (continuation target)
        self.brick_size = brick_size
        self.created_t_msc = created_t_msc
        self.filled = False
        self.fill_t_msc = None
        self.result = None
        self.exit_t_msc = None
        self.pnl_R = 0.0

def run_backtest(tick_path: Path):
    logger.info(f"Loading tick data from {tick_path}...")
    df = pd.read_parquet(tick_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    logger.info(f"Loaded {len(df):,} ticks across {len(days)} trading days.")

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
        if day not in daily_groups:
            continue
        day_df = daily_groups[day]
        day_ticks = day_df.to_dict('records')
        if len(day_ticks) < 100:
            continue

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
        brick_size = day_open * K_MULTIPLIER

        optimizer = PathOptimizer()
        if len(lookback_ticks) > 1000:
            best_price, _, _ = optimizer.find_optimal_anchor(lookback_ticks, brick_size)
            if best_price is None:
                best_price = day_open
        else:
            best_price = day_open

        renko = RenkoBuilder(best_price)
        renko.update_brick_size(brick_size, new_day_open=best_price)

        for tick in lookback_ticks:
            renko.update_tick(tick['bid'], tick['time_msc'])

        for tick in day_ticks:
            bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']

            # 1. Check pending orders for FILL
            still_pending = []
            for order in pending_orders:
                if (t_msc - order.created_t_msc) > ORDER_EXPIRY_MS:
                    order.result = 'expired'
                    all_orders.append(order)
                    continue

                filled = False
                if order.direction == 1:  # Buy limit
                    if ask <= order.limit_price:
                        filled = True
                elif order.direction == -1:  # Sell limit
                    if bid >= order.limit_price:
                        filled = True

                if filled:
                    order.filled = True
                    order.fill_t_msc = t_msc
                    active_trades.append(order)
                else:
                    still_pending.append(order)
            pending_orders = still_pending

            # 2. Check active trades for TP/SL
            still_active = []
            for trade in active_trades:
                if (t_msc - trade.fill_t_msc) > TRADE_TIMEOUT_MS:
                    trade.result = 'timeout'
                    trade.exit_t_msc = t_msc
                    trade.pnl_R = 0.0
                    all_orders.append(trade)
                    continue

                resolved = False
                if trade.direction == 1:  # Long fade
                    if bid >= trade.tp_price:
                        trade.result = 'win'
                        trade.pnl_R = +1.0
                        trade.exit_t_msc = t_msc
                        resolved = True
                    elif bid <= trade.sl_price:
                        trade.result = 'loss'
                        trade.pnl_R = -1.0
                        trade.exit_t_msc = t_msc
                        resolved = True
                elif trade.direction == -1:  # Short fade
                    if ask <= trade.tp_price:
                        trade.result = 'win'
                        trade.pnl_R = +1.0
                        trade.exit_t_msc = t_msc
                        resolved = True
                    elif ask >= trade.sl_price:
                        trade.result = 'loss'
                        trade.pnl_R = -1.0
                        trade.exit_t_msc = t_msc
                        resolved = True

                if resolved:
                    all_orders.append(trade)
                else:
                    still_active.append(trade)
            active_trades = still_active

            # 3. Check for new bricks → place new FADE orders
            new_bricks = renko.update_tick(bid, t_msc)
            for brick in new_bricks:
                order_counter += 1
                brick_dir = 1 if brick.uptrend == 1 else -1

                brick_open = brick.open
                brick_close = brick.close

                # Trade the Fade
                if brick_dir == 1:  # UP brick → Sell Limit at open
                    fade_dir = -1
                    limit_price = brick_open
                    tp_price = brick_open - 1 * brick_size  # +1R for fade
                    sl_price = brick_open + 1 * brick_size  # -1R for fade
                else:  # DOWN brick → Buy Limit at open
                    fade_dir = 1
                    limit_price = brick_open
                    tp_price = brick_open + 1 * brick_size  # +1R for fade
                    sl_price = brick_open - 1 * brick_size  # -1R for fade

                order = PendingOrder(
                    order_id=order_counter,
                    direction=fade_dir,
                    limit_price=limit_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    brick_size=brick_size,
                    created_t_msc=t_msc
                )
                pending_orders.append(order)

        if (day_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            resolved = len(all_orders)
            logger.info(f"  Day {day_idx+1}/{len(days)} ({day}) | "
                        f"Resolved: {resolved} | Pending: {len(pending_orders)} | "
                        f"Active: {len(active_trades)} | {elapsed:.1f}s")

    for trade in active_trades:
        trade.result = 'timeout'
        trade.pnl_R = 0.0
        all_orders.append(trade)
    for order in pending_orders:
        order.result = 'expired'
        all_orders.append(order)

    elapsed = time.time() - t0
    logger.info(f"\nBacktest complete in {elapsed:.1f}s")

    # ── Analysis ──
    total_orders = len(all_orders)
    filled = [o for o in all_orders if o.filled]
    expired = [o for o in all_orders if o.result == 'expired']
    wins = [o for o in all_orders if o.result == 'win']
    losses = [o for o in all_orders if o.result == 'loss']
    timeouts = [o for o in all_orders if o.result == 'timeout']

    logger.info(f"\n{'='*60}")
    logger.info(f"  FADE-AT-OPEN STRATEGY BACKTEST RESULTS")
    logger.info(f"{'='*60}")

    logger.info(f"\n  ── Order Flow ──")
    logger.info(f"  Total Orders Placed:    {total_orders:,}")
    logger.info(f"  Orders Filled:          {len(filled):,} ({len(filled)/total_orders*100:.1f}%)")
    logger.info(f"  Orders Expired:         {len(expired):,} ({len(expired)/total_orders*100:.1f}%)")

    if filled:
        logger.info(f"\n  ── Trade Results (Filled Only) ──")
        logger.info(f"  Wins (+1R):             {len(wins):,}")
        logger.info(f"  Losses (-1R):           {len(losses):,}")

        tradeable = len(wins) + len(losses)
        if tradeable > 0:
            win_rate = len(wins) / tradeable * 100
            loss_rate = len(losses) / tradeable * 100

            total_pnl_R = sum(o.pnl_R for o in all_orders)
            gross_profit = sum(o.pnl_R for o in wins)
            gross_loss = sum(o.pnl_R for o in losses)

            expectancy = total_pnl_R / tradeable
            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

            logger.info(f"\n  ── Key Metrics ──")
            logger.info(f"  Win Rate:               {win_rate:.2f}%")
            logger.info(f"  Loss Rate:              {loss_rate:.2f}%")
            logger.info(f"  Risk:Reward:            1:1 (Risk 1 to make 1)")
            logger.info(f"  Per-Trade Expectancy:   {expectancy:+.4f} R")
            logger.info(f"  Total P&L:              {total_pnl_R:+.1f} R")
            logger.info(f"  Profit Factor:          {profit_factor:.3f}")

            # ── Breakeven analysis ──
            breakeven_wr = 1.0 / (1.0 + 1.0) * 100  # For 1:1 R:R
            logger.info(f"\n  ── Breakeven Analysis ──")
            logger.info(f"  Breakeven WR for 1:1:            {breakeven_wr:.2f}%")
            logger.info(f"  Actual Win Rate:                 {win_rate:.2f}%")
            logger.info(f"  Edge:                            {win_rate - breakeven_wr:+.2f}%")

    logger.info(f"\n{'='*60}")

if __name__ == "__main__":
    tick_path = BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"
    run_backtest(tick_path)
