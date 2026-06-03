"""
Sim-Labeler V3 (Reversion Taxonomy)
===================================
Generates the path_class (0: Cont, 1: Pullback Cont, 2: Full Reversal) and 
reversion_depth labels. Also computes brick-level exhaustion features 
(wick_ratio, ofi_peak, velocity_peak) that cannot be derived from the 100-tick micro tensors.
Does NOT save .npy files to massively accelerate generation, as we will dynamically 
link to the V1 tensors.
"""

import os
import glob
import time
import logging
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import calendar

# ── Path Setup ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys_path_to_add = str(BASE_DIR / "BrickOfTicks_Trader")
import sys
if sys_path_to_add not in sys.path:
    sys.path.insert(0, sys_path_to_add)

import bridge.renko
import bridge.path_optimizer
bridge.renko.K_MULTIPLIER = 0.00118
bridge.path_optimizer.K_MULTIPLIER = 0.00118

from bridge.renko import RenkoBuilder, K_MULTIPLIER
from bridge.feature_engine import LiveFeatureEngine
from bridge.path_optimizer import PathOptimizer

logger = logging.getLogger(__name__)
OUTPUT_DIR = BASE_DIR / "outputs" / "sim_labels_v3"


class SimTrade_V3:
    def __init__(self, brick_id, timestamp, direction, entry, brick_size, brick_scalars, start_t_msc):
        self.brick_id = brick_id
        self.timestamp = timestamp
        self.direction = direction
        self.entry = entry
        self.brick_size = brick_size
        self.brick_scalars = brick_scalars # dict of wick_ratio, ofi_peak, etc.
        self.start_t_msc = start_t_msc
        
        self.touched_open = False
        self.lowest_price = entry
        self.highest_price = entry
        
        if self.direction == 1:
            self.target_cont = entry + brick_size
            self.target_open = entry - brick_size
            self.target_rev  = entry - 2 * brick_size
        else:
            self.target_cont = entry - brick_size
            self.target_open = entry + brick_size
            self.target_rev  = entry + 2 * brick_size

    def check_tick(self, bid, ask, t_msc):
        """Returns True if trade resolves, else False"""
        
        # 12-hour timeout (12 * 3600 * 1000 ms)
        if (t_msc - self.start_t_msc) > 43200000:
            self.path_class = -1
            self.reversion_depth = np.nan
            return True
        
        if self.direction == 1: # UP brick
            self.lowest_price = min(self.lowest_price, bid)
            
            if not self.touched_open:
                if bid >= self.target_cont: # 0: Immediate Continuation
                    self.path_class = 0
                    self.reversion_depth = (self.entry - self.lowest_price) / self.brick_size
                    return True
                elif bid <= self.target_open:
                    self.touched_open = True
                    
            if self.touched_open:
                if bid >= self.target_cont: # 1: Pullback Continuation
                    self.path_class = 1
                    self.reversion_depth = (self.entry - self.lowest_price) / self.brick_size
                    return True
                elif bid <= self.target_rev: # 2: Full Reversal
                    self.path_class = 2
                    self.reversion_depth = (self.entry - self.lowest_price) / self.brick_size
                    return True
                    
        else: # DOWN brick
            self.highest_price = max(self.highest_price, ask)
            
            if not self.touched_open:
                if ask <= self.target_cont: # 0: Immediate Continuation
                    self.path_class = 0
                    self.reversion_depth = (self.highest_price - self.entry) / self.brick_size
                    return True
                elif ask >= self.target_open:
                    self.touched_open = True
                    
            if self.touched_open:
                if ask <= self.target_cont: # 1: Pullback Continuation
                    self.path_class = 1
                    self.reversion_depth = (self.highest_price - self.entry) / self.brick_size
                    return True
                elif ask >= self.target_rev: # 2: Full Reversal
                    self.path_class = 2
                    self.reversion_depth = (self.highest_price - self.entry) / self.brick_size
                    return True
                    
        return False
        
    def to_dict(self, year, month):
        d = {
            "year": year,
            "month": month,
            "brick_id": self.brick_id,
            "timestamp": self.timestamp,
            "direction": self.direction,
            "path_class": getattr(self, 'path_class', np.nan),
            "reversion_depth": getattr(self, 'reversion_depth', np.nan)
        }
        d.update(self.brick_scalars)
        return d


class LabelingBroker_V3:
    def __init__(self, year, month):
        self.year = year
        self.month = month
        self.active_trades = []
        self.resolved_labels = []
        
    def open_trade(self, brick_id, timestamp, direction, entry, brick_size, brick_scalars, start_t_msc):
        self.active_trades.append(
            SimTrade_V3(brick_id, timestamp, direction, entry, brick_size, brick_scalars, start_t_msc)
        )
        
    def check_ticks(self, bid, ask, t_msc):
        remaining = []
        for trade in self.active_trades:
            resolved = trade.check_tick(bid, ask, t_msc)
            if resolved:
                self.resolved_labels.append(trade.to_dict(self.year, self.month))
            else:
                remaining.append(trade)
        self.active_trades = remaining


def process_chunk(year: int, month: int, files: list, target_start, target_end):
    log = logging.getLogger(f"Worker-{year}-{month:02d}")
    log.info(f"Processing chunk {year}-{month:02d}...")
    
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            pass
    if not dfs: return
        
    df = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in df.columns and 'time_msc' not in df.columns:
        df['time_msc'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
        
    for col in ['bid', 'ask', 'bid_vol', 'ask_vol']:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
            
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    
    optimizer = PathOptimizer()
    broker = LabelingBroker_V3(year, month)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    global_brick_id = 0
    total_active_days = 0
    
    for day_idx, day in enumerate(days):
        day_df = df[df['utc_day'] == day]
        day_ticks = day_df.to_dict('records')
        if len(day_ticks) < 100: continue
            
        is_warmup = day < target_start
        is_active = target_start <= day <= target_end
        is_closeout = day > target_end
        
        if is_warmup: continue
        
        if is_closeout:
            if not broker.active_trades: break 
            for tick in day_ticks:
                broker.check_ticks(tick['bid'], tick['ask'], tick['time_msc'])
            continue
            
        total_active_days += 1
        day_open = day_ticks[0]['bid']
        brick_size = day_open * K_MULTIPLIER
        
        start_date = day - pd.Timedelta(days=7)
        lb_mask = (df['utc_day'] >= start_date) & (df['utc_day'] < day)
        lb_df = df[lb_mask]
        
        best_price = day_open
        if len(lb_df) > 1000:
            best_price_opt, _, _ = optimizer.find_optimal_anchor(lb_df.to_dict('records'), brick_size)
            if best_price_opt is not None:
                best_price = best_price_opt
                
        renko = RenkoBuilder(best_price)
        renko.update_brick_size(brick_size, new_day_open=best_price)
        feature_engine = LiveFeatureEngine()
        feature_engine.update_brick_size(brick_size)
        
        # Track forming brick state for Exhaustion metrics
        brick_open_price = best_price
        curr_wick_low = float('inf')
        curr_wick_high = float('-inf')
        
        ofi_peak = 0.0
        ofi_history = []
        vel_peak = 0.0
        
        if len(lb_df) > 1000:
            lb_ticks = lb_df.to_dict('records')
            for tick in lb_ticks:
                bid_vol = 0.0 # Forced fallback for live matching
                ask_vol = 0.0 # Forced fallback for live matching
                feature_engine.compute_vector(tick['bid'], tick['ask'], bid_vol, ask_vol, tick['time_msc'])
                new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
                for brick in new_bricks:
                    feature_engine.on_new_brick(brick)
                    
        for tick in day_ticks:
            bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']
            bid_vol = 0.0 # Forced fallback for live matching
            ask_vol = 0.0 # Forced fallback for live matching
            
            feat_vec = feature_engine.compute_vector(bid, ask, bid_vol, ask_vol, t_msc)
            
            # Exhaustion tracking
            curr_wick_low = min(curr_wick_low, bid)
            curr_wick_high = max(curr_wick_high, ask)
            
            if feat_vec is not None and len(feat_vec) > 3:
                current_ofi = feat_vec[0]
                current_vel = feat_vec[3]
            else:
                current_ofi = 0.0
                current_vel = 0.0
                
            if abs(current_ofi) > abs(ofi_peak):
                ofi_peak = current_ofi
            ofi_history.append(current_ofi)
            
            vel_peak = max(vel_peak, current_vel)
            
            if is_active:
                broker.check_ticks(bid, ask, t_msc)
            
            new_bricks = renko.update_tick(bid, t_msc)
            for brick in new_bricks:
                feature_engine.on_new_brick(brick)
                
                if is_active:
                    global_brick_id += 1
                    
                    direction = 1 if brick.uptrend == 1 else -1
                    entry = ask if direction == 1 else bid
                    
                    # Compute wick ratio
                    if direction == 1:
                        wick_ratio = (brick_open_price - curr_wick_low) / brick_size
                    else:
                        wick_ratio = (curr_wick_high - brick_open_price) / brick_size
                        
                    # OFI Slope
                    ofi_slope = 0.0
                    if len(ofi_history) > 1:
                        x = np.arange(len(ofi_history))
                        y = np.array(ofi_history)
                        cov = np.sum((x - np.mean(x)) * (y - np.mean(y)))
                        var = np.sum((x - np.mean(x))**2)
                        ofi_slope = cov / var if var > 0 else 0.0
                    
                    brick_scalars = {
                        "wick_ratio": max(0.0, wick_ratio),
                        "ofi_peak": ofi_peak,
                        "ofi_current": current_ofi,
                        "ofi_exhaustion": ofi_peak - current_ofi,
                        "ofi_slope": ofi_slope,
                        "vel_peak": vel_peak,
                        "vel_current": current_vel,
                        "spread_current": ask - bid,
                        "absorption_index": abs(ofi_peak) / (vel_peak + 1.0)
                    }
                    
                    broker.open_trade(
                        brick_id=global_brick_id,
                        timestamp=brick.timestamp,
                        direction=direction,
                        entry=entry,
                        brick_size=brick.brick_size,
                        brick_scalars=brick_scalars,
                        start_t_msc=t_msc
                    )
                
                # Reset for next brick
                brick_open_price = entry
                curr_wick_low = bid
                curr_wick_high = ask
                ofi_peak = 0.0
                ofi_history = []
                vel_peak = 0.0
                    
        if is_active and total_active_days % 5 == 0:
            log.info(f"  [{year}-{month:02d}] Active Days: {total_active_days} | Bricks: {global_brick_id}")

    # Don't force close, just dump resolved
    labels_df = pd.DataFrame(broker.resolved_labels)
    if not labels_df.empty:
        # Drop NaNs and Timeouts (-1)
        labels_df = labels_df.dropna(subset=['path_class'])
        labels_df = labels_df[labels_df['path_class'] != -1]
        labels_df['path_class'] = labels_df['path_class'].astype(int)
        
        out_path = OUTPUT_DIR / f"v3_labels_{year}_{month:02d}.parquet"
        labels_df.to_parquet(out_path)
        log.info(f"✅ Saved {len(labels_df)} V3 labels to {out_path}")


def mp_worker(args):
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "sim_labels_v3"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()]
    )
    yr, mo, files, t_start, t_end = args
    process_chunk(yr, mo, files, t_start, t_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    all_files = glob.glob(str(BASE_DIR / "Data" / "Raw" / "Ticks" / "**" / "*.parquet"), recursive=True)
    
    file_dict = {}
    for f in all_files:
        path = Path(f)
        day_str, month_str, year_str = path.stem, path.parent.name, path.parent.parent.name
        if day_str.isdigit() and month_str.isdigit() and year_str.isdigit():
            try:
                dt = datetime(int(year_str), int(month_str), int(day_str)).date()
                file_dict[dt] = f
            except ValueError:
                pass
                
    year_months = sorted(list(set((dt.year, dt.month) for dt in file_dict.keys())))
    logger.info(f"Found {len(year_months)} unique months of data to process.")
    
    tasks = []
    for y, m in year_months:
        target_start = datetime(y, m, 1).date()
        _, last_day = calendar.monthrange(y, m)
        target_end = datetime(y, m, last_day).date()
        
        load_start = target_start - timedelta(days=7)
        load_end = target_end + timedelta(days=7)
        
        chunk_files = []
        curr = load_start
        while curr <= load_end:
            if curr in file_dict:
                chunk_files.append(file_dict[curr])
            curr += timedelta(days=1)
            
        if chunk_files:
            tasks.append((y, m, chunk_files, target_start, target_end))
    
    if args.workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=args.workers) as pool:
            pool.map(mp_worker, tasks)
    else:
        for t in tasks:
            mp_worker(t)

if __name__ == "__main__":
    main()
