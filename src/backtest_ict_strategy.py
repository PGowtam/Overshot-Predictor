import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
import collections

from ict_data_builder import ICTDataBuilder
from ict_core import StructureDetector, FVGDetector, find_order_block

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PIP = 0.10
# We assume a fixed $10,000 account for backtesting PnL tracking in terms of Risk units.
# R_multiple represents the raw risk multiple gained/lost.

class ActiveTrade:
    def __init__(self, trade_id, direction, entry, sl, tp1, tp2, risk_multiplier):
        self.trade_id = trade_id
        self.direction = direction  # 1 (Long), -1 (Short)
        self.entry_price = entry
        self.sl_price = sl
        self.initial_sl_price = sl
        self.tp1_price = tp1
        self.tp2_price = tp2
        self.risk_multiplier = risk_multiplier # 1.0 for full risk, 0.5 for half risk
        
        self.stage1_hit = False
        self.stage1_exit_price = None
        
        self.closed = False
        self.final_exit_price = None
        
        self.max_favorable_price = entry
        
    def update_max_excursion(self, price):
        if self.direction == 1:
            if price > self.max_favorable_price:
                self.max_favorable_price = price
        else:
            if price < self.max_favorable_price:
                self.max_favorable_price = price

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
        
        r_total = 0.0
        if self.stage1_hit:
            r_total = (r_stage1 * 0.5) + (r_stage2 * 0.5)
        else:
            r_total = r_stage2
            
        return r_total * self.risk_multiplier

def run_multi_year_backtest(tick_dir: Path):
    logger.info(f"Scanning for tick data in {tick_dir}...")
    
    if tick_dir.is_file():
        all_files = [tick_dir]
    else:
        all_files = list(tick_dir.glob("*/*/*.parquet"))
        all_files.sort(key=lambda x: str(x))
        
    if not all_files:
        logger.error("No parquet files found!")
        return

    logger.info(f"Found {len(all_files)} parquet files. Starting simulation...")

    builder = ICTDataBuilder()
    m5_struct = StructureDetector(left_bars=2, right_bars=2) # 5-bar fractal
    m5_fvg_det = FVGDetector()
    
    sweep_state = 0 
    sweep_extreme = None
    sweep_time = None
    
    pending_limit_dir = 0
    pending_limit_price = None
    pending_limit_sl = None
    pending_limit_tp1 = None
    pending_limit_tp2 = None
    pending_risk_mult = 1.0
    
    active_trade = None
    completed_trades = []
    trade_counter = 0
    
    t0 = time.time()
    total_ticks_processed = 0

    for file_idx, file_path in enumerate(all_files):
        try:
            df = pd.read_parquet(file_path)
            if 'time_msc' not in df.columns:
                if 'timestamp' in df.columns:
                    df['time_msc'] = df['timestamp'].astype(np.int64) // 1_000_000
                else:
                    continue
            df = df.sort_values('time_msc').reset_index(drop=True)
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            continue
            
        times = df['time_msc'].values
        bids = df['bid'].values
        asks = df['ask'].values
        vols = df['bid_vol'].values
        
        for i in range(len(times)):
            t_msc = times[i]
            bid = bids[i]
            ask = asks[i]
            vol = vols[i]
            
            m5_completed, m15_completed = builder.process_tick(t_msc, bid, vol)
            total_ticks_processed += 1
            
            # --- Trade Management ---
            if active_trade is not None:
                if active_trade.direction == 1:
                    active_trade.update_max_excursion(bid)
                else:
                    active_trade.update_max_excursion(ask)
                    
                if not active_trade.stage1_hit:
                    if active_trade.direction == 1 and bid >= active_trade.tp1_price:
                        active_trade.stage1_hit = True
                        active_trade.stage1_exit_price = active_trade.tp1_price
                        active_trade.sl_price = active_trade.entry_price # Move to BE
                    elif active_trade.direction == -1 and ask <= active_trade.tp1_price:
                        active_trade.stage1_hit = True
                        active_trade.stage1_exit_price = active_trade.tp1_price
                        active_trade.sl_price = active_trade.entry_price # Move to BE
                        
                if active_trade.direction == 1 and bid <= active_trade.sl_price:
                    active_trade.closed = True
                    active_trade.final_exit_price = active_trade.sl_price
                    completed_trades.append(active_trade)
                    active_trade = None
                elif active_trade.direction == -1 and ask >= active_trade.sl_price:
                    active_trade.closed = True
                    active_trade.final_exit_price = active_trade.sl_price
                    completed_trades.append(active_trade)
                    active_trade = None
                    
                if active_trade is not None:
                    if active_trade.direction == 1 and bid >= active_trade.tp2_price:
                        active_trade.closed = True
                        active_trade.final_exit_price = active_trade.tp2_price
                        completed_trades.append(active_trade)
                        active_trade = None
                    elif active_trade.direction == -1 and ask <= active_trade.tp2_price:
                        active_trade.closed = True
                        active_trade.final_exit_price = active_trade.tp2_price
                        completed_trades.append(active_trade)
                        active_trade = None

            # --- Limit Order Fill Check ---
            elif pending_limit_dir != 0:
                filled = False
                if pending_limit_dir == 1 and ask <= pending_limit_price:
                    filled = True
                    entry_p = pending_limit_price
                elif pending_limit_dir == -1 and bid >= pending_limit_price:
                    filled = True
                    entry_p = pending_limit_price
                    
                if filled:
                    trade_counter += 1
                    active_trade = ActiveTrade(
                        trade_id=trade_counter,
                        direction=pending_limit_dir,
                        entry=entry_p,
                        sl=pending_limit_sl,
                        tp1=pending_limit_tp1,
                        tp2=pending_limit_tp2,
                        risk_multiplier=pending_risk_mult
                    )
                    pending_limit_dir = 0
                    sweep_state = 0
                    
            # --- Strategy Logic (Bar Close) ---
            if m5_completed:
                swing = m5_struct.update(m5_completed)
                fvg = m5_fvg_det.update(m5_completed)
                
                in_kz = builder.session.in_london or builder.session.in_ny
                
                # GATE 0: HTF Trend Alignment
                val_d1_50 = builder.ema_d1_50.value
                val_h4_20 = builder.ema_h4_20.value
                
                htf_bullish = False
                htf_bearish = False
                
                if val_d1_50 is not None and val_h4_20 is not None:
                    if bid > val_d1_50 and bid > val_h4_20:
                        htf_bullish = True
                    elif bid < val_d1_50 and bid < val_h4_20:
                        htf_bearish = True
                        
                # GATE A: Sweep Detection
                if sweep_state == 0 and in_kz and active_trade is None and pending_limit_dir == 0:
                    pools = [
                        builder.session.ash, builder.session.asl,
                        builder.session.pdh, builder.session.pdl,
                        builder.session.london_h, builder.session.london_l
                    ]
                    pools = [p for p in pools if p is not None]
                    
                    if pools:
                        # Any sweep can be tracked, filtering happens in scoring
                        for p in pools:
                            if m5_completed.high > p and m5_completed.close < p:
                                sweep_state = -1
                                sweep_extreme = m5_completed.high
                                sweep_time = m5_completed.time_msc
                                break
                            elif m5_completed.low < p and m5_completed.close > p:
                                sweep_state = 1
                                sweep_extreme = m5_completed.low
                                sweep_time = m5_completed.time_msc
                                break
                                    
                # GATE B, C, D: CHoCH, Displacement, Entry
                if sweep_state != 0 and active_trade is None and pending_limit_dir == 0:
                    # Timeout sweep after 12 bars (1 hour)
                    if m5_completed.time_msc - sweep_time > 12 * 5 * 60 * 1000:
                        sweep_state = 0
                        
                    else:
                        if sweep_state == -1: # Bearish
                            recent_lows = [s for s in m5_struct.swings if not s.is_high]
                            if recent_lows:
                                last_low = recent_lows[-1].price
                                if m5_completed.close < last_low:
                                    # Gate D Enforcement
                                    body = abs(m5_completed.close - m5_completed.open)
                                    range_h_l = m5_completed.high - m5_completed.low
                                    body_pct = body / range_h_l if range_h_l > 0 else 0
                                    
                                    sma20_vol = sum(b.volume for b in builder.m5_history[-20:]) / 20.0 if len(builder.m5_history) >= 20 else 0
                                    
                                    if body_pct > 0.60 and m5_completed.volume > sma20_vol:
                                        ob = find_order_block(builder.m5_history, len(builder.m5_history)-1, is_bullish_impulse=False)
                                        
                                        # Calculate Confluence Score
                                        score = 2 # Sweep(1) + Kill Zone(1)
                                        if htf_bearish: score += 1
                                        d1_range = builder.session.current_day_h - builder.session.current_day_l
                                        if d1_range > 0:
                                            midpoint = builder.session.current_day_l + (d1_range / 2.0)
                                            if m5_completed.close > midpoint: # Premium
                                                score += 1
                                                
                                        if score >= 3:
                                            risk_mult = 1.0 if score >= 4 else 0.5
                                            if ob:
                                                pending_limit_dir = -1
                                                pending_limit_price = ob.midpoint
                                                pending_limit_sl = sweep_extreme + (6 * PIP)
                                                risk = abs(pending_limit_price - pending_limit_sl)
                                                pending_limit_tp1 = pending_limit_price - risk
                                                pending_limit_tp2 = pending_limit_price - (3 * risk)
                                                pending_risk_mult = risk_mult
                                            elif fvg and not fvg.is_bullish:
                                                pending_limit_dir = -1
                                                pending_limit_price = fvg.midpoint
                                                pending_limit_sl = sweep_extreme + (6 * PIP)
                                                risk = abs(pending_limit_price - pending_limit_sl)
                                                pending_limit_tp1 = pending_limit_price - risk
                                                pending_limit_tp2 = pending_limit_price - (3 * risk)
                                                pending_risk_mult = risk_mult
                                        
                        elif sweep_state == 1: # Bullish
                            recent_highs = [s for s in m5_struct.swings if s.is_high]
                            if recent_highs:
                                last_high = recent_highs[-1].price
                                if m5_completed.close > last_high:
                                    # Gate D Enforcement
                                    body = abs(m5_completed.close - m5_completed.open)
                                    range_h_l = m5_completed.high - m5_completed.low
                                    body_pct = body / range_h_l if range_h_l > 0 else 0
                                    
                                    sma20_vol = sum(b.volume for b in builder.m5_history[-20:]) / 20.0 if len(builder.m5_history) >= 20 else 0
                                    
                                    if body_pct > 0.60 and m5_completed.volume > sma20_vol:
                                        ob = find_order_block(builder.m5_history, len(builder.m5_history)-1, is_bullish_impulse=True)
                                        
                                        # Calculate Confluence Score
                                        score = 2 # Sweep(1) + Kill Zone(1)
                                        if htf_bullish: score += 1
                                        d1_range = builder.session.current_day_h - builder.session.current_day_l
                                        if d1_range > 0:
                                            midpoint = builder.session.current_day_l + (d1_range / 2.0)
                                            if m5_completed.close < midpoint: # Discount
                                                score += 1
                                                
                                        if score >= 3:
                                            risk_mult = 1.0 if score >= 4 else 0.5
                                            if ob:
                                                pending_limit_dir = 1
                                                pending_limit_price = ob.midpoint
                                                pending_limit_sl = sweep_extreme - (6 * PIP)
                                                risk = abs(pending_limit_price - pending_limit_sl)
                                                pending_limit_tp1 = pending_limit_price + risk
                                                pending_limit_tp2 = pending_limit_price + (3 * risk)
                                                pending_risk_mult = risk_mult
                                            elif fvg and fvg.is_bullish:
                                                pending_limit_dir = 1
                                                pending_limit_price = fvg.midpoint
                                                pending_limit_sl = sweep_extreme - (6 * PIP)
                                                risk = abs(pending_limit_price - pending_limit_sl)
                                                pending_limit_tp1 = pending_limit_price + risk
                                                pending_limit_tp2 = pending_limit_price + (3 * risk)
                                                pending_risk_mult = risk_mult

            # Cancel limit orders if end of session (11:30 NY time approx)
            if pending_limit_dir != 0:
                if builder.cached_ny_hour >= 11:
                    pending_limit_dir = 0
                    sweep_state = 0
                    
            # Close active trades before NY afternoon dead zone (16:00 EST)
            if active_trade is not None:
                if builder.cached_ny_hour >= 16:
                    active_trade.closed = True
                    active_trade.final_exit_price = bid if active_trade.direction == 1 else ask
                    completed_trades.append(active_trade)
                    active_trade = None

        if file_idx > 0 and file_idx % 20 == 0:
            elapsed = time.time() - t0
            logger.info(f"Processed {file_idx+1}/{len(all_files)} files ({total_ticks_processed:,} ticks) | "
                        f"Completed Trades: {len(completed_trades)} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    logger.info(f"\nMulti-year Backtest complete in {elapsed:.1f}s")
    
    total_trades = len(completed_trades)
    if total_trades > 0:
        r_multiples = [t.calculate_r_multiple() for t in completed_trades]
        wins = [r for r in r_multiples if r > 0]
        
        win_rate = len(wins) / total_trades * 100
        avg_r = sum(r_multiples) / total_trades
        
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Win Rate:     {win_rate:.1f}%")
        logger.info(f"  Total R:      {sum(r_multiples):+.2f} R")
        logger.info(f"  Avg Expect:   {avg_r:+.2f} R")
    else:
        logger.info("No trades executed.")

if __name__ == "__main__":
    tick_dir = Path(__file__).resolve().parent.parent / "Data" / "Raw" / "Ticks"
    if not tick_dir.exists():
        tick_dir = Path(__file__).resolve().parent.parent / "Data" / "xauusd_ticks_2026.parquet"
    run_multi_year_backtest(tick_dir)
