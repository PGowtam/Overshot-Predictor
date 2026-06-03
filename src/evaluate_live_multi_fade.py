"""
2026 Live Engine Simulation — Multi-Threshold Fade
========================================================
Runs the deep learning Baseline model across 2026 ticks using Multiprocessing.
Simultaneously evaluates three Fading thresholds (PW <= 0.35, 0.30, 0.20) using
three independent SimulatedBroker instances.
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
import calendar

# ── Path Setup ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TRADER_DIR = BASE_DIR / "BrickOfTicks_Trader"
sys.path.insert(0, str(TRADER_DIR))

import bridge.renko
import bridge.path_optimizer
bridge.renko.K_MULTIPLIER = 0.00118
bridge.path_optimizer.K_MULTIPLIER = 0.00118

from bridge.renko import RenkoBuilder, K_MULTIPLIER
from bridge.feature_engine import LiveFeatureEngine
from bridge.buffer import InferenceBuffer
from bridge.path_optimizer import PathOptimizer

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────
EXEC_DIR = BASE_DIR / "outputs" / "exec_baseline"
MODEL_PATH = EXEC_DIR / "model.keras"
OUTPUT_DIR = BASE_DIR / "outputs" / "experiments" / "sim_multi_fade_2026"

class SimulatedBroker:
    def __init__(self, name):
        self.name = name
        self.active_position = None
        self.trade_log = []
        self.next_ticket = 1000
        self.daily_pnl = 0.0
    
    def execute(self, direction, price, sl, tp, brick_size, timestamp):
        if self.active_position is not None:
            return None
        self.active_position = {
            "ticket": self.next_ticket,
            "direction": direction,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "brick_size": brick_size,
            "entry_time": timestamp
        }
        self.next_ticket += 1
        return {"status": "OK", "ticket": self.active_position["ticket"]}
    
    def check_tick(self, tick):
        pos = self.active_position
        if pos is None: return None
        
        if pos["direction"] == 1:
            if tick["bid"] <= pos["sl"]: return self._close("LOSS", pos["sl"], tick["time_msc"])
            if tick["bid"] >= pos["tp"]: return self._close("WIN", pos["tp"], tick["time_msc"])
        elif pos["direction"] == -1:
            if tick["ask"] >= pos["sl"]: return self._close("LOSS", pos["sl"], tick["time_msc"])
            if tick["ask"] <= pos["tp"]: return self._close("WIN", pos["tp"], tick["time_msc"])
        return None
    
    def force_close(self, tick):
        pos = self.active_position
        if pos is None: return None
        close_price = tick["bid"] if pos["direction"] == 1 else tick["ask"]
        return self._close("FORCE_CLOSE", close_price, tick["time_msc"])
    
    def _close(self, outcome, close_price, exit_time):
        pos = self.active_position
        pnl = (close_price - pos["entry"]) * pos["direction"]
        pnl_r = pnl / pos["brick_size"]
        
        record = {
            "broker_name": self.name,
            "ticket": pos["ticket"],
            "direction": pos["direction"],
            "entry": pos["entry"],
            "sl": pos["sl"],
            "tp": pos["tp"],
            "brick_size": pos["brick_size"],
            "outcome": outcome,
            "close_price": close_price,
            "pnl_pts": pnl,
            "pnl_r": pnl_r,
            "entry_time": pos["entry_time"],
            "exit_time": exit_time
        }
        self.trade_log.append(record)
        self.daily_pnl += pnl
        self.active_position = None
        return record
    
    def reset_daily(self):
        self.daily_pnl = 0.0
    
    @property
    def has_position(self):
        return self.active_position is not None

def process_month(year, month, parquet_path):
    """Worker process: Evaluates 3 thresholds simultaneously."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    
    print(f"[{year}-{month:02d}] Worker started.")
    
    target_start = pd.Timestamp(year, month, 1, tz="UTC")
    _, last_day = calendar.monthrange(year, month)
    target_end = pd.Timestamp(year, month, last_day, 23, 59, 59, tz="UTC")
    
    pad_start = target_start - pd.Timedelta(days=7)
    pad_end = target_end + pd.Timedelta(days=4)
    
    df = pd.read_parquet(parquet_path)
    df['time_msc_dt'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)
    df = df[(df['time_msc_dt'] >= pad_start) & (df['time_msc_dt'] <= pad_end)]
    if len(df) == 0:
        return {}, {}
        
    df['utc_day'] = df['time_msc_dt'].dt.date
    days = sorted(df['utc_day'].unique())
    
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    
    # Instantiate 3 independent brokers
    brokers = {
        "0.35": SimulatedBroker("PW_0.35"),
        "0.30": SimulatedBroker("PW_0.30"),
        "0.20": SimulatedBroker("PW_0.20")
    }
    
    optimizer = PathOptimizer()
    
    # Store daily summaries per broker
    daily_summaries = {"0.35": [], "0.30": [], "0.20": []}
    
    for day_idx, day in enumerate(days):
        day_df = df[df['utc_day'] == day]
        day_ticks = day_df.to_dict('records')
        if len(day_ticks) < 100: continue
        
        lb_df = df[(df['utc_day'] < day) & (df['utc_day'] >= day - pd.Timedelta(days=7))]
        all_history = lb_df.to_dict('records')
        
        day_open = day_ticks[0]['bid']
        brick_size = day_open * K_MULTIPLIER
        
        if len(all_history) > 1000:
            best_price, best_idx, _ = optimizer.find_optimal_anchor(all_history, brick_size)
            if best_price is None:
                best_price, best_idx = day_open, 0
        else:
            best_price, best_idx = day_open, 0
            
        renko = RenkoBuilder(best_price)
        renko.update_brick_size(brick_size, new_day_open=best_price)
        feature_engine = LiveFeatureEngine()
        feature_engine.update_brick_size(brick_size)
        buffer = InferenceBuffer()
        
        for b in brokers.values():
            b.reset_daily()
        
        # Warmup Phase
        warmup_bricks = 0
        for i, tick in enumerate(all_history):
            feat_vec = feature_engine.compute_vector(tick['bid'], tick['ask'], tick.get('bid_vol',0), tick.get('ask_vol',0), tick['time_msc'])
            if i >= best_idx:
                buffer.append_tick(feat_vec, renko.brick_count)
                for brick in renko.update_tick(tick['bid'], tick['time_msc']):
                    warmup_bricks += 1
                    feature_engine.on_new_brick(brick)
                    buffer.on_brick_close(renko.brick_count - 1, feature_engine.last_macro)
                    
        ticks_zscored = len(feature_engine.zs_ofi.deque)
        if warmup_bricks < 10 or ticks_zscored < 1000:
            continue
            
        is_active_month = (day.year == year and day.month == month)
        is_closeout_phase = (day.year > year or (day.year == year and day.month > month))
        
        day_bricks = 0
        
        day_trades_before = {k: len(b.trade_log) for k, b in brokers.items()}
        
        # Active Phase
        for tick in day_ticks:
            feat_vec = feature_engine.compute_vector(tick['bid'], tick['ask'], tick.get('bid_vol',0), tick.get('ask_vol',0), tick['time_msc'])
            buffer.append_tick(feat_vec, renko.brick_count)
            
            # Check open positions for all brokers independently
            for b in brokers.values():
                if b.has_position:
                    b.check_tick(tick)
                
            for brick in renko.update_tick(tick['bid'], tick['time_msc']):
                day_bricks += 1
                feature_engine.on_new_brick(brick)
                tensors = buffer.on_brick_close(renko.brick_count - 1, feature_engine.last_macro)
                
                if tensors is not None and is_active_month:
                    micro, macro = tensors
                    preds = model([micro, macro], training=False)
                    pw = float(preds[0].numpy().flatten()[0])
                    
                    bs = brick.brick_size
                    # Reversal Strategy: We trade OPPOSITE to the brick's direction
                    direction = -1 if brick.uptrend == 1 else 1
                    price = tick['bid'] if direction == -1 else tick['ask']
                    sl = price + bs if direction == -1 else price - bs
                    tp = price - bs if direction == -1 else price + bs
                    
                    # Evaluate each broker against its threshold
                    thresholds = {"0.35": 0.35, "0.30": 0.30, "0.20": 0.20}
                    for k, thresh in thresholds.items():
                        b = brokers[k]
                        if pw <= thresh and not b.has_position:
                            if b.daily_pnl >= -5.0 * bs:
                                b.execute(direction, price, sl, tp, bs, brick.timestamp)
        
        # Closeout End of Month
        for b in brokers.values():
            if b.has_position and is_closeout_phase:
                b.force_close(day_ticks[-1])
            elif b.has_position and len(day_ticks) > 0 and not is_active_month:
                b.force_close(day_ticks[-1])
            
        if is_active_month:
            for k, b in brokers.items():
                day_trades = b.trade_log[day_trades_before[k]:]
                wins = sum(1 for t in day_trades if t['outcome'] == 'WIN')
                losses = sum(1 for t in day_trades if t['outcome'] == 'LOSS')
                pnl_r = sum(t['pnl_r'] for t in day_trades)
                daily_summaries[k].append({
                    "day": str(day), "bricks": day_bricks, "trades": len(day_trades),
                    "wins": wins, "losses": losses, "pnl_r": round(pnl_r, 2)
                })
            
    print(f"[{year}-{month:02d}] Worker finished.")
    
    ret_logs = {k: b.trade_log for k, b in brokers.items()}
    return ret_logs, daily_summaries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=str, default=str(BASE_DIR / "data" / "xauusd_ticks_2026.parquet"))
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = [(2026, m, args.ticks) for m in range(1, 6)]
    
    print(f"🚀 Starting Multi-Threshold MP Simulation (Fade Model) with {min(8, len(tasks))} workers...")
    start_t = time.time()
    
    with mp.Pool(processes=min(8, len(tasks))) as pool:
        results = pool.starmap(process_month, tasks)
        
    all_trades = {"0.35": [], "0.30": [], "0.20": []}
    all_summaries = {"0.35": [], "0.30": [], "0.20": []}
    
    for month_logs, month_sums in results:
        for k in all_trades.keys():
            all_trades[k].extend(month_logs[k])
            all_summaries[k].extend(month_sums[k])
            
    print(f"\\n✅ Simulation completed in {(time.time()-start_t)/60:.1f} mins.")
    
    # Save CSVs
    for k in all_trades.keys():
        if all_trades[k]:
            pd.DataFrame(all_trades[k]).to_csv(OUTPUT_DIR / f"sim_trades_{k.replace('.', '')}.csv", index=False)
        if all_summaries[k]:
            pd.DataFrame(all_summaries[k]).to_csv(OUTPUT_DIR / f"sim_daily_summary_{k.replace('.', '')}.csv", index=False)
            
    # Generate Report
    report_lines = [
        "# 2026 Multi-Threshold Fade Strategy (Baseline Model)",
        "",
        "## Results Summary",
        "",
        "| Threshold | Total Trades | Wins | Losses | Win Rate | Gross PnL | EV per Trade |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for k in sorted(all_trades.keys(), reverse=True):
        trades = all_trades[k]
        total = len([t for t in trades if t['outcome'] in ('WIN', 'LOSS')])
        wins = sum(1 for t in trades if t['outcome'] == 'WIN')
        losses = sum(1 for t in trades if t['outcome'] == 'LOSS')
        pnl = sum(t['pnl_r'] for t in trades)
        wr = (wins / total * 100) if total > 0 else 0
        ev = (pnl / total) if total > 0 else 0
        
        report_lines.append(f"| PW <= {k} | {total} | {wins} | {losses} | **{wr:.2f}%** | {pnl:+.2f}R | {ev:+.4f}R |")
        
    report_lines.append("")
    report_lines.append(f"> Simulation generated concurrently in {(time.time()-start_t)/60:.1f} mins.")
    
    with open(OUTPUT_DIR / "multi_fade_report.md", "w") as f:
        f.write("\\n".join(report_lines))

if __name__ == "__main__":
    main()
