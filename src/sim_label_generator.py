"""
Execution-based Label Generator (Sim-Labeler)
=============================================
Generates y_class and y_mag labels by running an imaginary trade on every
single Renko brick using strict Bid/Ask execution prices, perfectly matching
the live simulation conditions at K=0.00118.
"""

import os
import glob
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

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
    def __init__(self):
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
                "brick_id": trade.brick_id,
                "timestamp": trade.timestamp,
                "direction": trade.direction,
                "y_class": y_class,
                "y_mag": y_mag
            })
        self.active_trades = []


def process_year(year: str, files: list):
    logger.info(f"Processing year {year} ({len(files)} files)...")
    
    # Load all parquet files for the year
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")
            
    if not dfs:
        return
        
    df = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in df.columns and 'time_msc' not in df.columns:
        df['time_msc'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
    df = df.sort_values('time_msc').reset_index(drop=True)
    
    # Convert back to UTC dates to detect boundaries
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    days = sorted(df['utc_day'].unique())
    
    optimizer = PathOptimizer()
    broker = LabelingBroker()
    
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    
    global_brick_id = 0
    total_days = len(days)
    
    logger.info(f"Year {year} has {len(df):,} ticks across {total_days} days.")
    
    for day_idx, day in enumerate(days):
        day_mask = df['utc_day'] == day
        day_df = df[day_mask]
        day_ticks = day_df.to_dict('records')
        
        if len(day_ticks) < 100:
            continue
            
        day_open = day_ticks[0]['bid']
        brick_size = day_open * K_MULTIPLIER
        
        # Path optimization over last 7 days
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
                
        # Run Renko & Pipeline
        renko = RenkoBuilder(best_price)
        renko.update_brick_size(brick_size, new_day_open=best_price)
        feature_engine = LiveFeatureEngine()
        feature_engine.update_brick_size(brick_size)
        buffer = InferenceBuffer()
        
        # ── Z-score Warmup ──
        if len(lb_df) > 1000:
            lb_ticks = lb_df.to_dict('records')
            for i, tick in enumerate(lb_ticks):
                # Pass actual bid_vol / ask_vol
                bid_vol = tick.get('bid_vol', 0.0)
                ask_vol = tick.get('ask_vol', 0.0)
                feat_vec = feature_engine.compute_vector(
                    tick['bid'], tick['ask'], bid_vol, ask_vol, tick['time_msc']
                )
                if best_price_opt is not None and i >= 0: # We should ideally start buffer appending after anchor
                    pass # Simplified warmup: just feed features for z-score. Buffer is short memory anyway.
                    # Actually, InferenceBuffer needs to be warmed up too, but since we are doing 7 days, it's fine.
        
        # We'll just reset feature engine and do full warmup like sim_live_engine_exec
        feature_engine = LiveFeatureEngine()
        feature_engine.update_brick_size(brick_size)
        buffer = InferenceBuffer()
        
        best_idx = 0 # Dummy for now, sim_live_engine uses exact start_idx from optimizer. We'll just use the day's ticks.
        # Let's do the proper warmup from lb_ticks:
        if len(lb_df) > 1000:
            lb_ticks = lb_df.to_dict('records')
            for i, tick in enumerate(lb_ticks):
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
                    
        # Now process today's ticks
        for tick in day_ticks:
            bid_vol = tick.get('bid_vol', 0.0)
            ask_vol = tick.get('ask_vol', 0.0)
            
            # 1. Compute features
            feat_vec = feature_engine.compute_vector(
                tick['bid'], tick['ask'], bid_vol, ask_vol, tick['time_msc']
            )
            
            # 2. Append to buffer
            buffer.append_tick(feat_vec, renko.brick_count)
            
            # 3. Check active trades
            broker.check_ticks(tick['bid'], tick['ask'])
            
            # 4. Form bricks
            new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
            for brick in new_bricks:
                global_brick_id += 1
                
                # Update feature engine and buffer
                feature_engine.on_new_brick(brick)
                buffer.on_brick_close(renko.brick_count - 1, feature_engine.last_macro)
                
                if len(buffer.snapshots) > 0:
                    micro = buffer.snapshots[-1]
                    macro = np.array(buffer.macro[-1], dtype=np.float32)
                    # Save tensors directly
                    np.save(TENSOR_DIR / f"micro_{global_brick_id}.npy", micro)
                    np.save(TENSOR_DIR / f"macro_{global_brick_id}.npy", macro)
                    
                    # Save Markov sequence
                    with open(TENSOR_DIR / f"seq_{global_brick_id}.txt", "w") as f:
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
                
        if (day_idx + 1) % 10 == 0:
            logger.info(f"  [{day_idx+1}/{total_days}] {day} | Bricks: {global_brick_id} | Active Trades: {len(broker.active_trades)}")

    # Force close at end of year
    if len(day_ticks) > 0:
        broker.force_close_all(day_ticks[-1]['bid'], day_ticks[-1]['ask'])
        
    labels_df = pd.DataFrame(broker.resolved_labels)
    if not labels_df.empty:
        out_path = OUTPUT_DIR / f"sim_labels_{year}.parquet"
        labels_df.to_parquet(out_path)
        logger.info(f"✅ Saved {len(labels_df)} labels for {year} to {out_path}")
        
        # Log distribution
        win_rate = labels_df['y_class'].mean() * 100
        logger.info(f"   Win Rate: {win_rate:.2f}%")
        logger.info(f"   LOSS y_mag mean: {labels_df[labels_df['y_class']==0]['y_mag'].mean():.4f}")
        logger.info(f"   WIN y_mag mean:  {labels_df[labels_df['y_class']==1]['y_mag'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="all", help="Specific year to process, or 'all'")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / "labeler.log"), mode='a')
        ]
    )
    
    all_files = glob.glob(str(BASE_DIR / "Data" / "Raw" / "Ticks" / "**" / "*.parquet"), recursive=True)
    
    # Group by year folder
    years = {}
    for f in all_files:
        parts = f.split(os.sep)
        # Typically .../Ticks/2023/05/20230501.parquet
        try:
            idx = parts.index("Ticks")
            yr = parts[idx+1]
            if yr.isdigit() and len(yr) == 4:
                if yr not in years:
                    years[yr] = []
                years[yr].append(f)
        except ValueError:
            pass
            
    sorted_years = sorted(years.keys())
    
    if args.year != "all":
        if args.year in years:
            sorted_years = [args.year]
        else:
            logger.error(f"Year {args.year} not found in data.")
            return

    for yr in sorted_years:
        process_year(yr, years[yr])
        
    logger.info("Label generation complete.")

if __name__ == "__main__":
    main()
