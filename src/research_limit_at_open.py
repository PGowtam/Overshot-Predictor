"""
Research: What distinguishes winning vs losing limit-at-open trades?
===================================================================
Captures STRUCTURAL / GEOMETRIC features for every trade (no microstructure):
  - Renko sequence pattern (streak length, reversal history)
  - Time-of-day / day-of-week
  - Intraday brick count (volatility proxy)
  - Spread-to-brick ratio (the only micro feature, but universal)
  - Distance from day open (trend strength)
  - Brick size regime
  - Sequence entropy
"""

import sys
import logging
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from collections import Counter

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
                 brick_size, created_t_msc, features):
        self.order_id = order_id
        self.direction = direction
        self.limit_price = limit_price
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.brick_size = brick_size
        self.created_t_msc = created_t_msc
        self.features = features  # dict of structural features
        self.filled = False
        self.fill_t_msc = None
        self.result = None
        self.exit_t_msc = None
        self.pnl_R = 0.0


def compute_sequence_entropy(seq, window=20):
    """Shannon entropy of the last N bits of the brick sequence."""
    if len(seq) < 2:
        return 0.0
    s = seq[-window:]
    counts = Counter(s)
    total = len(s)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_streak(seq):
    """How many consecutive same-direction bricks at the end of the sequence."""
    if not seq:
        return 0
    last = seq[-1]
    streak = 0
    for c in reversed(seq):
        if c == last:
            streak += 1
        else:
            break
    return streak


def compute_reversal_count(seq, window=20):
    """Number of direction changes in the last N bricks."""
    s = seq[-window:]
    if len(s) < 2:
        return 0
    changes = sum(1 for i in range(1, len(s)) if s[i] != s[i-1])
    return changes


def run_research(tick_path: Path):
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

        # Lookback
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

        # Warm up
        for tick in lookback_ticks:
            renko.update_tick(tick['bid'], tick['time_msc'])

        # Track intraday state
        intraday_brick_count = 0
        day_of_week = day.weekday()  # 0=Mon, 4=Fri

        for tick in day_ticks:
            bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']

            # Check pending fills
            still_pending = []
            for order in pending_orders:
                if (t_msc - order.created_t_msc) > ORDER_EXPIRY_MS:
                    order.result = 'expired'
                    all_orders.append(order)
                    continue
                filled = False
                if order.direction == 1 and ask <= order.limit_price:
                    filled = True
                elif order.direction == -1 and bid >= order.limit_price:
                    filled = True
                if filled:
                    order.filled = True
                    order.fill_t_msc = t_msc
                    active_trades.append(order)
                else:
                    still_pending.append(order)
            pending_orders = still_pending

            # Check active trades for TP/SL
            still_active = []
            for trade in active_trades:
                if (t_msc - trade.fill_t_msc) > TRADE_TIMEOUT_MS:
                    trade.result = 'timeout'
                    trade.pnl_R = 0.0
                    trade.exit_t_msc = t_msc
                    all_orders.append(trade)
                    continue
                resolved = False
                if trade.direction == 1:
                    if bid >= trade.tp_price:
                        trade.result = 'win'
                        trade.pnl_R = +2.0
                        trade.exit_t_msc = t_msc
                        resolved = True
                    elif bid <= trade.sl_price:
                        trade.result = 'loss'
                        trade.pnl_R = -1.0
                        trade.exit_t_msc = t_msc
                        resolved = True
                elif trade.direction == -1:
                    if ask <= trade.tp_price:
                        trade.result = 'win'
                        trade.pnl_R = +2.0
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

            # New bricks
            new_bricks = renko.update_tick(bid, t_msc)
            for brick in new_bricks:
                order_counter += 1
                intraday_brick_count += 1
                direction = 1 if brick.uptrend == 1 else -1

                brick_open = brick.open
                brick_close = brick.close

                if direction == 1:
                    limit_price = brick_open
                    tp_price = brick_open + 2 * brick_size
                    sl_price = brick_open - 1 * brick_size
                else:
                    limit_price = brick_open
                    tp_price = brick_open - 2 * brick_size
                    sl_price = brick_open + 1 * brick_size

                # ── Compute structural features ──
                seq = renko.sequence
                streak = compute_streak(seq)
                reversals_20 = compute_reversal_count(seq, 20)
                reversals_10 = compute_reversal_count(seq, 10)
                seq_entropy = compute_sequence_entropy(seq, 20)

                # Distance from day open (in brick sizes)
                dist_from_open = (bid - day_open) / brick_size

                # Hour of day (UTC)
                hour_utc = pd.Timestamp(t_msc, unit='ms', tz='UTC').hour

                # Spread-to-brick ratio
                spread = ask - bid
                spread_ratio = spread / brick_size

                # Last N pattern string (for pattern analysis)
                last_5 = seq[-5:] if len(seq) >= 5 else seq
                last_10 = seq[-10:] if len(seq) >= 10 else seq

                # Was the previous brick a reversal?
                is_after_reversal = 0
                if len(seq) >= 2:
                    is_after_reversal = 1 if seq[-1] != seq[-2] else 0

                features = {
                    'order_id': order_counter,
                    'direction': direction,
                    'brick_size': brick_size,
                    'day_of_week': day_of_week,
                    'hour_utc': hour_utc,
                    'streak': streak,
                    'reversals_20': reversals_20,
                    'reversals_10': reversals_10,
                    'seq_entropy': seq_entropy,
                    'dist_from_open': dist_from_open,
                    'intraday_brick_num': intraday_brick_count,
                    'spread_ratio': spread_ratio,
                    'spread': spread,
                    'is_after_reversal': is_after_reversal,
                    'last_5_pattern': last_5,
                    'last_10_pattern': last_10,
                    'total_bricks_in_seq': len(seq),
                }

                order = PendingOrder(
                    order_id=order_counter,
                    direction=direction,
                    limit_price=limit_price,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    brick_size=brick_size,
                    created_t_msc=t_msc,
                    features=features
                )
                pending_orders.append(order)

        if (day_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            logger.info(f"  Day {day_idx+1}/{len(days)} ({day}) | "
                        f"Resolved: {len(all_orders)} | {elapsed:.1f}s")

    # Force-close remaining
    for trade in active_trades:
        trade.result = 'timeout'
        trade.pnl_R = 0.0
        all_orders.append(trade)
    for order in pending_orders:
        order.result = 'expired'
        all_orders.append(order)

    elapsed = time.time() - t0
    logger.info(f"\nBacktest complete in {elapsed:.1f}s")

    # ── Build analysis DataFrame ──
    filled = [o for o in all_orders if o.result in ('win', 'loss')]
    logger.info(f"Analyzing {len(filled)} resolved trades (win/loss only)...")

    rows = []
    for o in filled:
        row = o.features.copy()
        row['result'] = o.result
        row['pnl_R'] = o.pnl_R
        row['is_win'] = 1 if o.result == 'win' else 0
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # ── ANALYSIS ──
    print("\n" + "="*70)
    print("  STRUCTURAL FEATURE ANALYSIS: WINNERS vs LOSERS")
    print("="*70)

    numeric_features = [
        'streak', 'reversals_20', 'reversals_10', 'seq_entropy',
        'dist_from_open', 'intraday_brick_num', 'spread_ratio',
        'is_after_reversal', 'hour_utc', 'day_of_week'
    ]

    wins = results_df[results_df['is_win'] == 1]
    losses = results_df[results_df['is_win'] == 0]

    print(f"\n  Total Wins:   {len(wins):,}")
    print(f"  Total Losses: {len(losses):,}")
    print(f"  Base WR:      {len(wins)/(len(wins)+len(losses))*100:.2f}%")

    print(f"\n  {'Feature':<25} {'Win Mean':>10} {'Loss Mean':>10} {'Delta':>10} {'Win/Loss':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for feat in numeric_features:
        w_mean = wins[feat].mean()
        l_mean = losses[feat].mean()
        delta = w_mean - l_mean
        ratio = w_mean / l_mean if l_mean != 0 else float('inf')
        print(f"  {feat:<25} {w_mean:>10.4f} {l_mean:>10.4f} {delta:>+10.4f} {ratio:>10.3f}")

    # ── Win rate by streak length ──
    print(f"\n  ── Win Rate by STREAK LENGTH ──")
    print(f"  {'Streak':<10} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Exp(R)':>8}")
    for s in sorted(results_df['streak'].unique()):
        subset = results_df[results_df['streak'] == s]
        wr = subset['is_win'].mean() * 100
        exp = subset['pnl_R'].mean()
        print(f"  {s:<10} {len(subset):>8} {subset['is_win'].sum():>8} {wr:>7.2f}% {exp:>+7.4f}")

    # ── Win rate by hour (UTC) ──
    print(f"\n  ── Win Rate by HOUR (UTC) ──")
    print(f"  {'Hour':<10} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Exp(R)':>8}")
    for h in sorted(results_df['hour_utc'].unique()):
        subset = results_df[results_df['hour_utc'] == h]
        wr = subset['is_win'].mean() * 100
        exp = subset['pnl_R'].mean()
        print(f"  {h:<10} {len(subset):>8} {subset['is_win'].sum():>8} {wr:>7.2f}% {exp:>+7.4f}")

    # ── Win rate by day of week ──
    print(f"\n  ── Win Rate by DAY OF WEEK ──")
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    print(f"  {'Day':<10} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Exp(R)':>8}")
    for d in sorted(results_df['day_of_week'].unique()):
        subset = results_df[results_df['day_of_week'] == d]
        wr = subset['is_win'].mean() * 100
        exp = subset['pnl_R'].mean()
        print(f"  {dow_names.get(d, str(d)):<10} {len(subset):>8} {subset['is_win'].sum():>8} {wr:>7.2f}% {exp:>+7.4f}")

    # ── Win rate by is_after_reversal ──
    print(f"\n  ── Win Rate by IS_AFTER_REVERSAL ──")
    print(f"  {'After Rev?':<12} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Exp(R)':>8}")
    for v in [0, 1]:
        subset = results_df[results_df['is_after_reversal'] == v]
        wr = subset['is_win'].mean() * 100
        exp = subset['pnl_R'].mean()
        label = "Yes" if v == 1 else "No"
        print(f"  {label:<12} {len(subset):>8} {subset['is_win'].sum():>8} {wr:>7.2f}% {exp:>+7.4f}")

    # ── Win rate by spread ratio bins ──
    print(f"\n  ── Win Rate by SPREAD RATIO (spread/brick_size) ──")
    bins = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 1.0]
    results_df['spread_bin'] = pd.cut(results_df['spread_ratio'], bins=bins)
    print(f"  {'Spread Bin':<20} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Exp(R)':>8}")
    for bin_label in results_df['spread_bin'].cat.categories:
        subset = results_df[results_df['spread_bin'] == bin_label]
        if len(subset) > 0:
            wr = subset['is_win'].mean() * 100
            exp = subset['pnl_R'].mean()
            print(f"  {str(bin_label):<20} {len(subset):>8} {subset['is_win'].sum():>8} {wr:>7.2f}% {exp:>+7.4f}")

    # ── Win rate by distance from day open (binned) ──
    print(f"\n  ── Win Rate by DISTANCE FROM DAY OPEN (in brick sizes) ──")
    dist_bins = [-100, -10, -5, -3, -1, 0, 1, 3, 5, 10, 100]
    results_df['dist_bin'] = pd.cut(results_df['dist_from_open'], bins=dist_bins)
    print(f"  {'Dist Bin':<20} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Exp(R)':>8}")
    for bin_label in results_df['dist_bin'].cat.categories:
        subset = results_df[results_df['dist_bin'] == bin_label]
        if len(subset) > 0:
            wr = subset['is_win'].mean() * 100
            exp = subset['pnl_R'].mean()
            print(f"  {str(bin_label):<20} {len(subset):>8} {subset['is_win'].sum():>8} {wr:>7.2f}% {exp:>+7.4f}")

    # ── Win rate by intraday brick count bins ──
    print(f"\n  ── Win Rate by INTRADAY BRICK COUNT (volatility proxy) ──")
    vol_bins = [0, 5, 10, 20, 30, 50, 100, 500]
    results_df['vol_bin'] = pd.cut(results_df['intraday_brick_num'], bins=vol_bins)
    print(f"  {'Brick Count':<20} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Exp(R)':>8}")
    for bin_label in results_df['vol_bin'].cat.categories:
        subset = results_df[results_df['vol_bin'] == bin_label]
        if len(subset) > 0:
            wr = subset['is_win'].mean() * 100
            exp = subset['pnl_R'].mean()
            print(f"  {str(bin_label):<20} {len(subset):>8} {subset['is_win'].sum():>8} {wr:>7.2f}% {exp:>+7.4f}")

    # ── Sequence entropy ──
    print(f"\n  ── Win Rate by SEQUENCE ENTROPY (20-brick window) ──")
    ent_bins = [0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]
    results_df['ent_bin'] = pd.cut(results_df['seq_entropy'], bins=ent_bins)
    print(f"  {'Entropy Bin':<20} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Exp(R)':>8}")
    for bin_label in results_df['ent_bin'].cat.categories:
        subset = results_df[results_df['ent_bin'] == bin_label]
        if len(subset) > 0:
            wr = subset['is_win'].mean() * 100
            exp = subset['pnl_R'].mean()
            print(f"  {str(bin_label):<20} {len(subset):>8} {subset['is_win'].sum():>8} {wr:>7.2f}% {exp:>+7.4f}")

    # ── Top-5 and Bottom-5 last_5 patterns ──
    print(f"\n  ── Win Rate by LAST-5-BRICK PATTERN (top & bottom) ──")
    pattern_stats = []
    for pat, grp in results_df.groupby('last_5_pattern'):
        if len(grp) >= 30:  # min sample size
            wr = grp['is_win'].mean() * 100
            exp = grp['pnl_R'].mean()
            pattern_stats.append((pat, len(grp), grp['is_win'].sum(), wr, exp))

    pattern_stats.sort(key=lambda x: x[3], reverse=True)
    print(f"  {'Pattern':<12} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Exp(R)':>8}")
    print(f"  --- TOP 5 ---")
    for pat, n, w, wr, exp in pattern_stats[:5]:
        print(f"  {pat:<12} {n:>8} {w:>8} {wr:>7.2f}% {exp:>+7.4f}")
    print(f"  --- BOTTOM 5 ---")
    for pat, n, w, wr, exp in pattern_stats[-5:]:
        print(f"  {pat:<12} {n:>8} {w:>8} {wr:>7.2f}% {exp:>+7.4f}")

    # ── COMBINED FILTER TEST ──
    print(f"\n{'='*70}")
    print(f"  FILTER EXPLORATION")
    print(f"{'='*70}")

    # Test combinations
    filters = [
        ("streak >= 2", results_df['streak'] >= 2),
        ("streak >= 3", results_df['streak'] >= 3),
        ("streak == 1 (reversal entry)", results_df['streak'] == 1),
        ("is_after_reversal == 1", results_df['is_after_reversal'] == 1),
        ("is_after_reversal == 0 (continuation)", results_df['is_after_reversal'] == 0),
        ("entropy < 0.7 (trending)", results_df['seq_entropy'] < 0.7),
        ("entropy >= 0.85 (choppy)", results_df['seq_entropy'] >= 0.85),
        ("spread_ratio < 0.06", results_df['spread_ratio'] < 0.06),
        ("spread_ratio < 0.04", results_df['spread_ratio'] < 0.04),
        ("intraday_brick < 10 (quiet)", results_df['intraday_brick_num'] < 10),
        ("intraday_brick 10-30 (moderate)", (results_df['intraday_brick_num'] >= 10) & (results_df['intraday_brick_num'] <= 30)),
        ("reversals_20 <= 3 (trending)", results_df['reversals_20'] <= 3),
        ("reversals_20 >= 8 (choppy)", results_df['reversals_20'] >= 8),
        ("streak >= 2 & spread < 0.06", (results_df['streak'] >= 2) & (results_df['spread_ratio'] < 0.06)),
        ("streak >= 3 & spread < 0.06", (results_df['streak'] >= 3) & (results_df['spread_ratio'] < 0.06)),
        ("streak >= 2 & entropy < 0.7", (results_df['streak'] >= 2) & (results_df['seq_entropy'] < 0.7)),
        ("streak >= 3 & entropy < 0.7", (results_df['streak'] >= 3) & (results_df['seq_entropy'] < 0.7)),
        ("streak >= 2 & rev_20 <= 3", (results_df['streak'] >= 2) & (results_df['reversals_20'] <= 3)),
        ("after_rev & spread < 0.06", (results_df['is_after_reversal'] == 1) & (results_df['spread_ratio'] < 0.06)),
        ("streak >= 2 & brick < 20", (results_df['streak'] >= 2) & (results_df['intraday_brick_num'] < 20)),
    ]

    print(f"\n  {'Filter':<40} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'Exp(R)':>8} {'TotalR':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, mask in filters:
        subset = results_df[mask]
        if len(subset) > 0:
            wr = subset['is_win'].mean() * 100
            exp = subset['pnl_R'].mean()
            total_r = subset['pnl_R'].sum()
            print(f"  {name:<40} {len(subset):>8} {subset['is_win'].sum():>8} {wr:>7.2f}% {exp:>+7.4f} {total_r:>+7.1f}")

    # Save raw data for further analysis
    out_path = BASE_DIR / "outputs" / "limit_at_open_research.parquet"
    results_df.to_parquet(out_path, index=False)
    logger.info(f"\nSaved {len(results_df)} trade records to {out_path}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    tick_path = BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"
    run_research(tick_path)
