import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
import collections
import itertools

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
        self.direction = direction
        self.entry_price = entry_price
        self.entry_t_msc = entry_t_msc
        self.sl_price = sl_price
        self.initial_sl_price = sl_price
        self.brick_size = brick_size
        self.tp1_price = tp1_price
        self.time_stop_msc = time_stop_msc
        
        self.stage1_hit = False
        self.stage1_exit_price = None
        
        self.closed = False
        self.final_exit_price = None
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
    df['datetime'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)
    df_dt = df.set_index('datetime')
    
    daily_bars = df_dt['bid'].resample('1d').ohlc()
    daily_bars.dropna(inplace=True)
    
    prev_close = daily_bars['close'].shift(1)
    tr1 = daily_bars['high'] - daily_bars['low']
    tr2 = (daily_bars['high'] - prev_close).abs()
    tr3 = (daily_bars['low'] - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    daily_atr = atr.shift(1)
    
    h4_bars = df_dt['bid'].resample('4h').last()
    h4_bars.dropna(inplace=True)
    
    ema20_4h = h4_bars.ewm(span=20, adjust=False).mean()
    ema20_4h_slope = ema20_4h.diff()
    h4_slope = ema20_4h_slope.shift(1)
    
    return daily_atr, h4_slope

class StrategyState:
    def __init__(self, entry_bricks, tp1_mult):
        self.entry_bricks = entry_bricks
        self.tp1_mult = tp1_mult
        
        self.completed_trades = []
        self.active_trade = None
        self.trade_counter = 0
        self.pending_entry_direction = 0
        self.pending_entry_brick_extreme = 0.0

def main():
    tick_path = BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"
    if not tick_path.exists():
        tick_path = BASE_DIR / "Data" / "xauusd_ticks_5ers_2026.parquet"
        
    logger.info(f"Loading data from {tick_path}...")
    df = pd.read_parquet(tick_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    
    logger.info("Preparing HTF Data...")
    daily_atr, h4_slope = prepare_htf_data(df)
    
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    
    # TRUNCATE TO 60 DAYS FOR SWEEP
    sweep_days = days[:60]
    df = df[df['utc_day'].isin(sweep_days)]
    
    logger.info(f"Loaded {len(df):,} ticks across {len(sweep_days)} trading days for sweep.")

    daily_groups = {}
    for day, group in df.groupby('utc_day'):
        daily_groups[day] = group

    multipliers = [0.8, 1.0, 1.2]
    entry_bricks_list = [2, 3]
    tp1_mults = [2, 3]
    
    results = []
    
    t0_total = time.time()
    
    for mult in multipliers:
        t0_pass = time.time()
        logger.info(f"\n--- Starting Pass for ATR Multiplier: {mult}x ---")
        
        # Create the 4 concurrent strategy states for this multiplier
        states = []
        for eb in entry_bricks_list:
            for tp in tp1_mults:
                states.append(StrategyState(eb, tp))
                
        # Shared Renko & Indicator State
        renko = None
        ema9 = EMA(9)
        ema21 = EMA(21)
        recent_directions = collections.deque(maxlen=3) # max lookback needed is 3
        recent_tick_counts = collections.deque(maxlen=20)
        recent_durations = collections.deque(maxlen=20)
        
        current_brick_tick_count = 0
        last_brick_t_msc = None
        
        for day_idx, day in enumerate(sweep_days):
            if day not in daily_groups:
                continue
                
            ts_day = pd.Timestamp(day, tz='UTC')
            if ts_day not in daily_atr.index or pd.isna(daily_atr[ts_day]):
                continue
                
            atr_14 = daily_atr[ts_day]
            brick_size = atr_14 * mult
            
            day_df = daily_groups[day]
            day_ticks = day_df.to_dict('records')
            if len(day_ticks) < 100:
                continue

            day_open = day_ticks[0]['bid']

            if renko is None:
                renko = RenkoBuilder(day_open)
                renko.update_brick_size(brick_size, new_day_open=day_open)
                
                # Warmup
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
                            recent_directions.append(1 if brick.uptrend else -1)
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
                
                # 1. Manage Entries & Active Trades for all 4 states
                for st in states:
                    # Entries
                    if st.pending_entry_direction != 0 and st.active_trade is None:
                        st.trade_counter += 1
                        entry_price = ask if st.pending_entry_direction == 1 else bid
                        
                        if st.pending_entry_direction == 1:
                            sl_price = st.pending_entry_brick_extreme - (1.5 * brick_size)
                            tp1_price = entry_price + (st.tp1_mult * brick_size)
                        else:
                            sl_price = st.pending_entry_brick_extreme + (1.5 * brick_size)
                            tp1_price = entry_price - (st.tp1_mult * brick_size)
                            
                        avg_duration = np.mean(recent_durations) if len(recent_durations) > 0 else 3600000
                        time_stop_msc = t_msc + (3 * avg_duration)
                            
                        st.active_trade = ActiveTrade(st.trade_counter, st.pending_entry_direction, entry_price, t_msc, sl_price, brick_size, tp1_price, time_stop_msc)
                        st.pending_entry_direction = 0
                        
                    # Trade Management
                    if st.active_trade is not None:
                        if st.active_trade.direction == 1:
                            st.active_trade.update_max_excursion(bid)
                        else:
                            st.active_trade.update_max_excursion(ask)
                        
                        # TP1 Check
                        if not st.active_trade.stage1_hit:
                            if st.active_trade.direction == 1 and bid >= st.active_trade.tp1_price:
                                st.active_trade.stage1_hit = True
                                st.active_trade.stage1_exit_price = st.active_trade.tp1_price
                                st.active_trade.sl_price = st.active_trade.entry_price
                            elif st.active_trade.direction == -1 and ask <= st.active_trade.tp1_price:
                                st.active_trade.stage1_hit = True
                                st.active_trade.stage1_exit_price = st.active_trade.tp1_price
                                st.active_trade.sl_price = st.active_trade.entry_price
                                
                        # Time Stop Check
                        if not st.active_trade.stage1_hit and t_msc > st.active_trade.time_stop_msc:
                            dist_to_tp1 = abs(st.active_trade.tp1_price - (bid if st.active_trade.direction == 1 else ask))
                            if dist_to_tp1 > (0.3 * st.active_trade.brick_size):
                                st.active_trade.closed = True
                                st.active_trade.final_exit_price = bid if st.active_trade.direction == 1 else ask
                                st.active_trade.exit_reason = "time_stop"
                                st.completed_trades.append(st.active_trade)
                                st.active_trade = None

                        # SL / Trailing Check
                        if st.active_trade is not None:
                            current_sl = st.active_trade.get_trailing_sl()
                            if st.active_trade.direction == 1 and bid <= current_sl:
                                st.active_trade.closed = True
                                st.active_trade.final_exit_price = current_sl
                                st.active_trade.exit_reason = "sl"
                                st.completed_trades.append(st.active_trade)
                                st.active_trade = None
                            elif st.active_trade.direction == -1 and ask >= current_sl:
                                st.active_trade.closed = True
                                st.active_trade.final_exit_price = current_sl
                                st.active_trade.exit_reason = "sl"
                                st.completed_trades.append(st.active_trade)
                                st.active_trade = None

                # 2. Update Shared Renko State
                new_bricks = renko.update_tick(bid, t_msc)
                for brick in new_bricks:
                    ema9.update(brick.close)
                    ema21.update(brick.close)
                    recent_directions.append(1 if brick.uptrend else -1)
                    recent_tick_counts.append(current_brick_tick_count)
                    if last_brick_t_msc is not None:
                        recent_durations.append(brick.timestamp - last_brick_t_msc)
                    last_brick_t_msc = brick.timestamp
                    
                    # 3. Check for New Entries for each state
                    for st in states:
                        if st.active_trade is None and st.pending_entry_direction == 0:
                            if len(recent_directions) >= st.entry_bricks and len(recent_tick_counts) >= 5 and ema9.value is not None and ema21.value is not None:
                                avg_vol = np.mean(recent_tick_counts)
                                vol_ok = current_brick_tick_count >= (1.1 * avg_vol)
                                
                                long_trend = ema9.value > ema21.value
                                long_price_above = brick.close > ema9.value and brick.close > ema21.value
                                long_bricks = all(d == 1 for d in list(recent_directions)[-st.entry_bricks:])
                                
                                short_trend = ema9.value < ema21.value
                                short_price_below = brick.close < ema9.value and brick.close < ema21.value
                                short_bricks = all(d == -1 for d in list(recent_directions)[-st.entry_bricks:])
                                
                                dt_tick = pd.to_datetime(t_msc, unit='ms', utc=True)
                                h4_bucket = dt_tick.floor('4h')
                                htf_slope = h4_slope.get(h4_bucket, 0.0)
                                if pd.isna(htf_slope): htf_slope = 0.0

                                if long_trend and long_price_above and long_bricks and vol_ok and (htf_slope > 0):
                                    st.pending_entry_direction = 1
                                    st.pending_entry_brick_extreme = brick.low
                                elif short_trend and short_price_below and short_bricks and vol_ok and (htf_slope < 0):
                                    st.pending_entry_direction = -1
                                    st.pending_entry_brick_extreme = brick.high

                    current_brick_tick_count = 0

            # End of day timeouts
            for st in states:
                if st.active_trade is not None:
                     if (t_msc - st.active_trade.entry_t_msc) > (48 * 3600 * 1000):
                         st.active_trade.closed = True
                         st.active_trade.final_exit_price = bid if st.active_trade.direction == 1 else ask
                         st.active_trade.exit_reason = "hard_timeout"
                         st.completed_trades.append(st.active_trade)
                         st.active_trade = None

            if (day_idx + 1) % 10 == 0:
                logger.info(f"    Completed {day_idx+1}/{len(sweep_days)} days...")

        # End of pass cleanup
        for st in states:
            if st.active_trade is not None:
                st.active_trade.closed = True
                st.active_trade.final_exit_price = bid if st.active_trade.direction == 1 else ask
                st.completed_trades.append(st.active_trade)
            
            # Save results
            total_trades = len(st.completed_trades)
            if total_trades > 0:
                r_multiples = [t.calculate_r_multiple() for t in st.completed_trades]
                wins = [r for r in r_multiples if r > 0]
                win_rate = len(wins) / total_trades * 100
                total_r = sum(r_multiples)
                avg_r = total_r / total_trades
            else:
                win_rate = 0.0
                total_r = 0.0
                avg_r = 0.0
                
            results.append({
                'Mult': mult,
                'Entry_Bricks': st.entry_bricks,
                'TP1_Mult': st.tp1_mult,
                'Total_Trades': total_trades,
                'Win_Rate': win_rate,
                'Total_R': total_r,
                'Avg_R': avg_r
            })
            
        logger.info(f"  Pass complete in {time.time() - t0_pass:.1f}s")
            
    elapsed = time.time() - t0_total
    logger.info(f"\nAll passes complete in {elapsed:.1f}s")
    
    # Sort results by Total R descending
    results.sort(key=lambda x: x['Total_R'], reverse=True)
    
    print("\n" + "="*85)
    print(f"{'ATR Mult':<10} | {'Entry Brk':<10} | {'TP1 Mult':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Total R':<10} | {'Avg R'}")
    print("-" * 85)
    for res in results:
        print(f"{res['Mult']:<10.1f} | {res['Entry_Bricks']:<10d} | {res['TP1_Mult']:<10d} | {res['Total_Trades']:<8d} | {res['Win_Rate']:<9.1f}% | {res['Total_R']:<+9.2f} R | {res['Avg_R']:<+6.3f} R")
    print("="*85)

if __name__ == "__main__":
    main()
