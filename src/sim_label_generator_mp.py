"""
Execution-based Label Generator (Sim-Labeler)
=============================================
Generates y_class and y_mag labels by running an imaginary trade on every
single Renko brick. Features an 8-worker Month-Level chunking architecture
with a 3-Phase approach (Warmup, Active, Closeout) for max efficiency.
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
from bridge.buffer import InferenceBuffer
from bridge.path_optimizer import PathOptimizer

logger = logging.getLogger(__name__)
OUTPUT_DIR = BASE_DIR / "outputs" / "sim_labels"
TENSOR_DIR = OUTPUT_DIR / "tensors"


class SimTrade:
    def __init__(self, brick_id, timestamp, direction, entry, brick_size):
        self.brick_id = brick_id
        self.timestamp = timestamp
        self.direction = direction
        self.entry = entry
        self.brick_size = brick_size
        self.peak = entry
        self.tp_hit = False
        
        if self.direction == 1:
            self.tp = entry + brick_size
            self.sl = entry - brick_size
        else:
            self.tp = entry - brick_size
            self.sl = entry + brick_size
            
    def check_tick(self, bid, ask):
        """Returns (outcome, y_mag) if trade resolves, else None"""
        if self.direction == 1:  # LONG
            self.peak = max(self.peak, bid)
            
            if not self.tp_hit:
                if bid >= self.tp:
                    self.tp_hit = True
                elif bid <= self.sl:
                    y_mag = abs(self.peak - self.entry) / self.brick_size
                    return (0, y_mag)
                    
            if self.tp_hit:
                if bid <= self.peak - self.brick_size or bid <= self.sl:
                    y_mag = abs(self.peak - self.entry) / self.brick_size
                    return (1, y_mag)
                    
        else:  # SHORT
            self.peak = min(self.peak, ask)
            
            if not self.tp_hit:
                if ask <= self.tp:
                    self.tp_hit = True
                elif ask >= self.sl:
                    y_mag = abs(self.peak - self.entry) / self.brick_size
                    return (0, y_mag)
                    
            if self.tp_hit:
                if ask >= self.peak + self.brick_size or ask >= self.sl:
                    y_mag = abs(self.peak - self.entry) / self.brick_size
                    return (1, y_mag)
                    
        return None
        
    def force_close(self, bid, ask):
        """Force close at end of data."""
        if self.direction == 1:
            self.peak = max(self.peak, bid)
        else:
            self.peak = min(self.peak, ask)
            
        y_mag = abs(self.peak - self.entry) / self.brick_size
        y_class = 1 if self.tp_hit else 0
        return (y_class, y_mag)


class LabelingBroker:
    def __init__(self, year, month):
        self.year = year
        self.month = month
        self.active_trades = []
        self.resolved_labels = []
        
    def open_trade(self, brick_id, timestamp, direction, entry, brick_size):
        self.active_trades.append(
            SimTrade(brick_id, timestamp, direction, entry, brick_size)
        )
        
    def check_ticks(self, bid, ask):
        remaining = []
        for trade in self.active_trades:
            result = trade.check_tick(bid, ask)
            if result is not None:
                y_class, y_mag = result
                self.resolved_labels.append({
                    "year": self.year,
                    "month": self.month,
                    "brick_id": trade.brick_id,
                    "timestamp": trade.timestamp,
                    "direction": trade.direction,
                    "y_class": y_class,
                    "y_mag": y_mag
                })
            else:
                remaining.append(trade)
        self.active_trades = remaining
        
    def force_close_all(self, bid, ask):
        for trade in self.active_trades:
            y_class, y_mag = trade.force_close(bid, ask)
            self.resolved_labels.append({
                "year": self.year,
                "month": self.month,
                "brick_id": trade.brick_id,
                "timestamp": trade.timestamp,
                "direction": trade.direction,
                "y_class": y_class,
                "y_mag": y_mag
            })
        self.active_trades = []


def process_chunk(year: int, month: int, files: list, target_start, target_end):
    log = logging.getLogger(f"Worker-{year}-{month:02d}")
    log.info(f"Processing chunk {year}-{month:02d} ({len(files)} padded files)...")
    
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            log.error(f"Error reading {f}: {e}")
            
    if not dfs:
        return
        
    df = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in df.columns and 'time_msc' not in df.columns:
        df['time_msc'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
        
    # Memory Optimization: Downcast to float32
    for col in ['bid', 'ask', 'bid_vol', 'ask_vol']:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
            
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    
    optimizer = PathOptimizer()
    broker = LabelingBroker(year, month)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    
    global_brick_id = 0
    total_active_days = 0
    
    for day_idx, day in enumerate(days):
        day_df = df[df['utc_day'] == day]
        day_ticks = day_df.to_dict('records')
        
        if len(day_ticks) < 100:
            continue
            
        is_warmup = day < target_start
        is_active = target_start <= day <= target_end
        is_closeout = day > target_end
        
        if is_warmup:
            continue
            
        # ── Phase 3: Closeout (Check existing trades ONLY) ──
        if is_closeout:
            if not broker.active_trades:
                break # All trades resolved! We can safely terminate processing this chunk.
                
            for tick in day_ticks:
                broker.check_ticks(tick['bid'], tick['ask'])
            continue
            
        # ── Phase 1 & 2: Active ──
        total_active_days += 1
            
        day_open = day_ticks[0]['bid']
        brick_size = day_open * K_MULTIPLIER
        
        start_date = day - pd.Timedelta(days=7)
        lb_mask = (df['utc_day'] >= start_date) & (df['utc_day'] < day)
        lb_df = df[lb_mask]
        
        best_price = day_open
        if len(lb_df) > 1000:
            best_price_opt, _, _ = optimizer.find_optimal_anchor(
                lb_df.to_dict('records'), brick_size
            )
            if best_price_opt is not None:
                best_price = best_price_opt
                
        renko = RenkoBuilder(best_price)
        renko.update_brick_size(brick_size, new_day_open=best_price)
        feature_engine = LiveFeatureEngine()
        feature_engine.update_brick_size(brick_size)
        buffer = InferenceBuffer()
        
        if len(lb_df) > 1000:
            lb_ticks = lb_df.to_dict('records')
            for tick in lb_ticks:
                bid_vol = tick.get('bid_vol', 0.0)
                ask_vol = tick.get('ask_vol', 0.0)
                feat_vec = feature_engine.compute_vector(
                    tick['bid'], tick['ask'], bid_vol, ask_vol, tick['time_msc']
                )
                buffer.append_tick(feat_vec, renko.brick_count)
                new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
                for brick in new_bricks:
                    feature_engine.on_new_brick(brick)
                    buffer.on_brick_close(renko.brick_count - 1, feature_engine.last_macro)
                    
        for tick in day_ticks:
            bid_vol = tick.get('bid_vol', 0.0)
            ask_vol = tick.get('ask_vol', 0.0)
            
            feat_vec = feature_engine.compute_vector(
                tick['bid'], tick['ask'], bid_vol, ask_vol, tick['time_msc']
            )
            buffer.append_tick(feat_vec, renko.brick_count)
            
            if is_active:
                broker.check_ticks(tick['bid'], tick['ask'])
            
            new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
            for brick in new_bricks:
                feature_engine.on_new_brick(brick)
                buffer.on_brick_close(renko.brick_count - 1, feature_engine.last_macro)
                
                if is_active:
                    global_brick_id += 1
                    
                    if len(buffer.snapshots) > 0:
                        micro = buffer.snapshots[-1]
                        macro = np.array(buffer.macro[-1], dtype=np.float32)
                        
                        # Save tensors
                        np.save(TENSOR_DIR / f"micro_{year}_{month:02d}_{global_brick_id}.npy", micro)
                        np.save(TENSOR_DIR / f"macro_{year}_{month:02d}_{global_brick_id}.npy", macro)
                        with open(TENSOR_DIR / f"seq_{year}_{month:02d}_{global_brick_id}.txt", "w") as f:
                            f.write(brick.sequence)
                    
                    # Open imaginary trade
                    direction = 1 if brick.uptrend == 1 else -1
                    entry = tick['ask'] if direction == 1 else tick['bid']
                    broker.open_trade(
                        brick_id=global_brick_id,
                        timestamp=brick.timestamp,
                        direction=direction,
                        entry=entry,
                        brick_size=brick.brick_size
                    )
                    
        if is_active and total_active_days % 5 == 0:
            log.info(f"  [{year}-{month:02d}] Processed {total_active_days} days | Bricks: {global_brick_id} | Active Trades: {len(broker.active_trades)}")

    # Force close any stragglers
    if broker.active_trades and len(df) > 0:
        last_row = df.iloc[-1]
        broker.force_close_all(last_row['bid'], last_row['ask'])
        
    labels_df = pd.DataFrame(broker.resolved_labels)
    if not labels_df.empty:
        out_path = OUTPUT_DIR / f"sim_labels_{year}_{month:02d}.parquet"
        labels_df.to_parquet(out_path)
        log.info(f"✅ Saved {len(labels_df)} labels for {year}-{month:02d} to {out_path}")


def mp_worker(args):
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "sim_labels"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / "labeler_mp.log"), mode='a')
        ]
    )
    yr, mo, files, t_start, t_end = args
    process_chunk(yr, mo, files, t_start, t_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8, help="Number of MP workers")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / "labeler_mp.log"), mode='a')
        ]
    )
    
    logger.info("Discovering all parquet files...")
    all_files = glob.glob(str(BASE_DIR / "Data" / "Raw" / "Ticks" / "**" / "*.parquet"), recursive=True)
    
    file_dict = {}
    for f in all_files:
        path = Path(f)
        day_str = path.stem
        month_str = path.parent.name
        year_str = path.parent.parent.name
        
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
        logger.info(f"Starting multiprocessing pool with {args.workers} workers...")
        with mp.Pool(processes=args.workers) as pool:
            pool.map(mp_worker, tasks)
    else:
        for t in tasks:
            mp_worker(t)
        
    logger.info("Label generation complete.")

if __name__ == "__main__":
    main()
