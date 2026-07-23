import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
import collections

BASE_DIR = Path(__file__).resolve().parent.parent
TRADER_DIR = BASE_DIR / "BrickOfTicks_Trader"
sys.path.insert(0, str(TRADER_DIR))

import bridge.renko
import bridge.path_optimizer

from bridge.renko import RenkoBuilder
from bridge.path_optimizer import PathOptimizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class EMA:
    def __init__(self, period):
        self.period = period
        self.alpha = 2 / (period + 1)
        self.value = None

    def update(self, price):
        if self.value is None:
            self.value = price
        else:
            self.value = (price * self.alpha) + (self.value * (1 - self.alpha))
        return self.value

class ActiveTrade:
    def __init__(self, trade_id, direction, entry_price, entry_t_msc, sl_price, brick_size, tp1_price, time_stop_msc):
        self.trade_id = trade_id
        self.direction = direction  # 1 (Long), -1 (Short)
        self.entry_price = entry_price
        self.entry_t_msc = entry_t_msc
        self.sl_price = sl_price
        self.initial_sl_price = sl_price
        self.brick_size = brick_size
        self.tp1_price = tp1_price
        self.time_stop_msc = time_stop_msc # Absolute timestamp when trade should timeout
        
        self.stage1_hit = False
        self.stage1_exit_price = None
        self.stage1_exit_t_msc = None
        
        self.closed = False
        self.final_exit_price = None
        self.final_exit_t_msc = None
        self.exit_reason = ""
        
        self.max_favorable_price = entry_price
        
    def update_max_excursion(self, price):
        if self.direction == 1:
            if price > self.max_favorable_price:
                self.max_favorable_price = price
        else:
            if price < self.max_favorable_price:
                self.max_favorable_price = price
                
    def get_trailing_sl(self):
        if not self.stage1_hit:
            return self.sl_price
        
        # Stage 2 Trailing SL: 2 brick sizes from max favorable price
        if self.direction == 1:
            return self.max_favorable_price - (2 * self.brick_size)
        else:
            return self.max_favorable_price + (2 * self.brick_size)
            
    def calculate_r_multiple(self):
        initial_risk = abs(self.entry_price - self.initial_sl_price)
        if initial_risk == 0:
            return 0.0
            
        r_stage1 = 0.0
        if self.stage1_hit:
            raw_pnl1 = (self.stage1_exit_price - self.entry_price) * self.direction
            r_stage1 = raw_pnl1 / initial_risk
            
        raw_pnl2 = (self.final_exit_price - self.entry_price) * self.direction
        r_stage2 = raw_pnl2 / initial_risk
        
        if self.stage1_hit:
            return (r_stage1 * 0.5) + (r_stage2 * 0.5)
        else:
            return r_stage2

def prepare_htf_data(df):
    logger.info("Preparing Daily ATR(14) and 4H EMA(20)...")
    
    # Needs a datetime index
    df['datetime'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)
    df_dt = df.set_index('datetime')
    
    # 1. Daily ATR(14)
    # Using 'bid' for OHLC
    daily_bars = df_dt['bid'].resample('1d').ohlc()
    daily_bars.dropna(inplace=True)
    
    # True Range
    # TR = max(H-L, |H-Cp|, |L-Cp|)
    prev_close = daily_bars['close'].shift(1)
    tr1 = daily_bars['high'] - daily_bars['low']
    tr2 = (daily_bars['high'] - prev_close).abs()
    tr3 = (daily_bars['low'] - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(14) - simple moving average of TR
    atr = tr.rolling(window=14).mean()
    
    # Shift by 1 so today's ATR is based on yesterday's closed data
    daily_atr = atr.shift(1)
    
    # 2. 4H EMA(20)
    h4_bars = df_dt['bid'].resample('4h').last()
    h4_bars.dropna(inplace=True)
    
    ema20_4h = h4_bars.ewm(span=20, adjust=False).mean()
    # Slope = current EMA - previous EMA
    ema20_4h_slope = ema20_4h.diff()
    
    # Shift by 1 so the current 4H bucket uses the previous closed 4H bucket's slope
    h4_slope = ema20_4h_slope.shift(1)
    
    return daily_atr, h4_slope

def run_backtest(tick_path: Path):
    logger.info(f"Loading tick data from {tick_path}...")
    df = pd.read_parquet(tick_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    
    daily_atr, h4_slope = prepare_htf_data(df)
    
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    logger.info(f"Loaded {len(df):,} ticks across {len(days)} trading days.")

    daily_groups = {}
    for day, group in df.groupby('utc_day'):
        daily_groups[day] = group

    completed_trades = []
    active_trade = None
    trade_counter = 0

    t0 = time.time()
    
    pending_entry_direction = 0
    pending_entry_brick_extreme = 0.0

    # Initialize continuous state
    renko = None
    ema9 = EMA(9)
    ema21 = EMA(21)
    recent_directions = collections.deque(maxlen=3)
    recent_tick_counts = collections.deque(maxlen=20)
    recent_durations = collections.deque(maxlen=20)
    
    current_brick_tick_count = 0
    last_brick_t_msc = None

    for day_idx, day in enumerate(days):
        if day not in daily_groups:
            continue
            
        # Get ATR for the day
        ts_day = pd.Timestamp(day, tz='UTC')
        if ts_day not in daily_atr.index or pd.isna(daily_atr[ts_day]):
            continue # Need ATR to proceed
            
        atr_14 = daily_atr[ts_day]
        brick_size = atr_14 * 1.2
        
        day_df = daily_groups[day]
        day_ticks = day_df.to_dict('records')
        if len(day_ticks) < 100:
            continue

        day_open = day_ticks[0]['bid']

        if renko is None:
            renko = RenkoBuilder(day_open)
            renko.update_brick_size(brick_size, new_day_open=day_open)
            # Fast-forward 20 days to warmup
            start_lb = day - timedelta(days=20)
            lb_dfs = []
            curr = start_lb
            while curr < day:
                if curr in daily_groups:
                    lb_dfs.append(daily_groups[curr])
                curr += timedelta(days=1)
            
            if lb_dfs:
                lb_df = pd.concat(lb_dfs, ignore_index=True)
                for tick in lb_df.to_dict('records'):
                    current_brick_tick_count += 1
                    new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
                    for brick in new_bricks:
                        ema9.update(brick.close)
                        ema21.update(brick.close)
                        recent_directions.append(brick.uptrend)
                        recent_tick_counts.append(current_brick_tick_count)
                        if last_brick_t_msc is not None:
                            recent_durations.append(brick.timestamp - last_brick_t_msc)
                        last_brick_t_msc = brick.timestamp
                        current_brick_tick_count = 0
        else:
            renko.update_brick_size(brick_size)

        for tick in day_ticks:
            bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']
            current_brick_tick_count += 1
            
            # 1. Handle Pending Entries
            if pending_entry_direction != 0 and active_trade is None:
                trade_counter += 1
                
                # Execute Market Order
                entry_price = ask if pending_entry_direction == 1 else bid
                
                if pending_entry_direction == 1:
                    sl_price = pending_entry_brick_extreme - (1.5 * brick_size)
                    tp1_price = entry_price + (3 * brick_size)
                else:
                    sl_price = pending_entry_brick_extreme + (1.5 * brick_size)
                    tp1_price = entry_price - (3 * brick_size)
                    
                # Time Stop calculation
                avg_duration = np.mean(recent_durations) if len(recent_durations) > 0 else 3600000 # default 1h
                time_stop_msc = t_msc + (3 * avg_duration)
                    
                active_trade = ActiveTrade(
                    trade_id=trade_counter,
                    direction=pending_entry_direction,
                    entry_price=entry_price,
                    entry_t_msc=t_msc,
                    sl_price=sl_price,
                    brick_size=brick_size,
                    tp1_price=tp1_price,
                    time_stop_msc=time_stop_msc
                )
                pending_entry_direction = 0
                
            # 2. Manage Active Trade
            if active_trade is not None:
                if active_trade.direction == 1:
                    active_trade.update_max_excursion(bid)
                else:
                    active_trade.update_max_excursion(ask)
                
                # Check Stage 1 TP
                if not active_trade.stage1_hit:
                    if active_trade.direction == 1 and bid >= active_trade.tp1_price:
                        active_trade.stage1_hit = True
                        active_trade.stage1_exit_price = active_trade.tp1_price
                        active_trade.stage1_exit_t_msc = t_msc
                        active_trade.sl_price = active_trade.entry_price
                    elif active_trade.direction == -1 and ask <= active_trade.tp1_price:
                        active_trade.stage1_hit = True
                        active_trade.stage1_exit_price = active_trade.tp1_price
                        active_trade.stage1_exit_t_msc = t_msc
                        active_trade.sl_price = active_trade.entry_price
                        
                # Check Time Stop (only if stage 1 not hit)
                if not active_trade.stage1_hit and t_msc > active_trade.time_stop_msc:
                    # Exception: Do not apply time stop if price is within 0.3 bricks of Stage 1 target
                    dist_to_tp1 = abs(active_trade.tp1_price - (bid if active_trade.direction == 1 else ask))
                    if dist_to_tp1 > (0.3 * active_trade.brick_size):
                        active_trade.closed = True
                        active_trade.final_exit_price = bid if active_trade.direction == 1 else ask
                        active_trade.final_exit_t_msc = t_msc
                        active_trade.exit_reason = "time_stop"
                        completed_trades.append(active_trade)
                        active_trade = None

                # Check SL or Trailing SL
                if active_trade is not None:
                    current_sl = active_trade.get_trailing_sl()
                    if active_trade.direction == 1 and bid <= current_sl:
                        active_trade.closed = True
                        active_trade.final_exit_price = current_sl # approximate
                        active_trade.final_exit_t_msc = t_msc
                        active_trade.exit_reason = "sl"
                        completed_trades.append(active_trade)
                        active_trade = None
                    elif active_trade.direction == -1 and ask >= current_sl:
                        active_trade.closed = True
                        active_trade.final_exit_price = current_sl
                        active_trade.final_exit_t_msc = t_msc
                        active_trade.exit_reason = "sl"
                        completed_trades.append(active_trade)
                        active_trade = None

            # 3. Update Renko & Strategy Logic
            new_bricks = renko.update_tick(bid, t_msc)
            for brick in new_bricks:
                ema9.update(brick.close)
                ema21.update(brick.close)
                recent_directions.append(1 if brick.uptrend else -1)
                recent_tick_counts.append(current_brick_tick_count)
                
                if last_brick_t_msc is not None:
                    recent_durations.append(brick.timestamp - last_brick_t_msc)
                last_brick_t_msc = brick.timestamp
                
                if active_trade is None and pending_entry_direction == 0:
                    # Require at least 20 bricks of history for averages, but we might only have 10-15 if market is slow.
                    # Relaxing strict len(recent_tick_counts) == 20 to len >= 5 for safety, but use whatever we have for avg.
                    if len(recent_directions) == 3 and len(recent_tick_counts) >= 5 and ema9.value is not None and ema21.value is not None:
                        avg_vol = np.mean(recent_tick_counts)
                        current_vol = current_brick_tick_count
                        
                        vol_ok = current_vol >= (1.1 * avg_vol)
                        
                        long_trend = ema9.value > ema21.value
                        long_price_above = brick.close > ema9.value and brick.close > ema21.value
                        long_3_bricks = all(d == 1 for d in list(recent_directions)[-3:])
                        
                        short_trend = ema9.value < ema21.value
                        short_price_below = brick.close < ema9.value and brick.close < ema21.value
                        short_3_bricks = all(d == -1 for d in list(recent_directions)[-3:])
                        
                        # Fetch 4H EMA Slope
                        dt_tick = pd.to_datetime(t_msc, unit='ms', utc=True)
                        # Round down to nearest 4H boundary for lookup
                        h4_bucket = dt_tick.floor('4h')
                        
                        htf_slope = 0.0
                        if h4_bucket in h4_slope.index:
                            htf_slope = h4_slope[h4_bucket]
                            if pd.isna(htf_slope):
                                htf_slope = 0.0

                        if long_trend and long_price_above and long_3_bricks and vol_ok and (htf_slope > 0):
                            pending_entry_direction = 1
                            pending_entry_brick_extreme = brick.low
                        elif short_trend and short_price_below and short_3_bricks and vol_ok and (htf_slope < 0):
                            pending_entry_direction = -1
                            pending_entry_brick_extreme = brick.high

                current_brick_tick_count = 0

        # Close trades over weekend if needed, but we'll let them run up to 48h total
        if active_trade is not None:
             if (t_msc - active_trade.entry_t_msc) > (48 * 3600 * 1000):
                 active_trade.closed = True
                 active_trade.final_exit_price = bid if active_trade.direction == 1 else ask
                 active_trade.final_exit_t_msc = t_msc
                 active_trade.exit_reason = "hard_timeout"
                 completed_trades.append(active_trade)
                 active_trade = None

        if (day_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info(f"  Day {day_idx+1}/{len(days)} ({day}) | "
                        f"Completed Trades: {len(completed_trades)} | {elapsed:.1f}s")

    # Close any remaining active trade
    if active_trade is not None:
        active_trade.closed = True
        active_trade.final_exit_price = bid if active_trade.direction == 1 else ask
        active_trade.final_exit_t_msc = t_msc
        active_trade.exit_reason = "eod"
        completed_trades.append(active_trade)

    elapsed = time.time() - t0
    logger.info(f"\nBacktest complete in {elapsed:.1f}s")

    # ── Analysis ──
    total_trades = len(completed_trades)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  FULL RENKO STRATEGY BACKTEST RESULTS")
    logger.info(f"{'='*60}")

    if total_trades > 0:
        r_multiples = [t.calculate_r_multiple() for t in completed_trades]
        wins = [r for r in r_multiples if r > 0]
        losses = [r for r in r_multiples if r <= 0]
        
        stage1_hits = [t for t in completed_trades if t.stage1_hit]
        time_stops = [t for t in completed_trades if t.exit_reason == "time_stop"]
        
        win_rate = len(wins) / total_trades * 100
        total_r = sum(r_multiples)
        avg_r = total_r / total_trades
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = sum(losses) if losses else 0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Max consecutive losses
        max_consec_losses = 0
        current_consec = 0
        for r in r_multiples:
            if r <= 0:
                current_consec += 1
                max_consec_losses = max(max_consec_losses, current_consec)
            else:
                current_consec = 0

        logger.info(f"\n  ── Trade Results ──")
        logger.info(f"  Total Trades Taken:     {total_trades:,}")
        logger.info(f"  Stage 1 Hits (Partial): {len(stage1_hits):,} ({len(stage1_hits)/total_trades*100:.1f}%)")
        logger.info(f"  Time Stops Hit:         {len(time_stops):,} ({len(time_stops)/total_trades*100:.1f}%)")
        logger.info(f"  Profitable Trades:      {len(wins):,}")
        logger.info(f"  Losing Trades:          {len(losses):,}")

        logger.info(f"\n  ── Key Metrics ──")
        logger.info(f"  Win Rate:               {win_rate:.2f}%")
        logger.info(f"  Average Win:            {avg_win:+.2f} R")
        logger.info(f"  Average Loss:           {avg_loss:+.2f} R")
        logger.info(f"  Avg R:R Ratio:          1:{abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Avg R:R Ratio:          N/A")
        logger.info(f"  Per-Trade Expectancy:   {avg_r:+.4f} R")
        logger.info(f"  Total P&L:              {total_r:+.2f} R")
        logger.info(f"  Profit Factor:          {profit_factor:.3f}")
        logger.info(f"  Max Consec Losses:      {max_consec_losses}")
    else:
        logger.info("  No trades taken.")

    logger.info(f"\n{'='*60}")

if __name__ == "__main__":
    tick_path = BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"
    if not tick_path.exists():
        tick_path = BASE_DIR / "Data" / "xauusd_ticks_5ers_2026.parquet"
    run_backtest(tick_path)
