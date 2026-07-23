"""
Track A5: 2026 Live Rule Execution (Fixed)
==========================================
Tests the "High Spread & Low OFI" rule out-of-sample on the 2026 tick dataset.
Fixes applied:
1. Allows concurrent overlapping trades to match the statistical study.
2. Exact matching of OFI peak and spread_current generation logic.
3. Logs first 100 signals for feature validation against v3 labels.
"""

import os
import sys
import math
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
TRADER_DIR = BASE_DIR / "BrickOfTicks_Trader"
sys.path.insert(0, str(TRADER_DIR))

import bridge.renko
import bridge.path_optimizer
bridge.renko.K_MULTIPLIER = 0.00118
bridge.path_optimizer.K_MULTIPLIER = 0.00118

from bridge.renko import RenkoBuilder, K_MULTIPLIER
from bridge.feature_engine import LiveFeatureEngine
from bridge.path_optimizer import PathOptimizer

logger = logging.getLogger(__name__)

OUTPUT_DIR = BASE_DIR / "outputs" / "experiments" / "rule_test_2026"

# ══════════════════════════════════════════════════════════════════
#  Simulation-Only Classes
# ══════════════════════════════════════════════════════════════════

class TickReplayProvider:
    def __init__(self, parquet_path: str):
        logger.info(f"Loading tick data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.df = self.df.sort_values('time_msc').reset_index(drop=True)
        self.df['utc_day'] = pd.to_datetime(self.df['time_msc'], unit='ms', utc=True).dt.date
        self.days = sorted(self.df['utc_day'].unique())
        
    def get_day_ticks(self, day) -> list:
        mask = self.df['utc_day'] == day
        return self.df[mask].to_dict('records')
    
    def get_lookback_ticks(self, target_day, lookback_days=7) -> list:
        target_idx = self.days.index(target_day)
        start_idx = max(0, target_idx - lookback_days)
        lookback_days_list = self.days[start_idx:target_idx]
        if not lookback_days_list: return []
        mask = self.df['utc_day'].isin(lookback_days_list)
        return self.df[mask].to_dict('records')


class SimTrade_V3_Broker:
    def __init__(self):
        self.active_positions = []
        self.trade_log = []
        self.next_ticket = 1000
    
    def execute(self, direction, price, sl, tp, brick_size, timestamp):
        pos = {
            "ticket": self.next_ticket,
            "direction": direction,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "brick_size": brick_size,
            "entry_time": timestamp
        }
        self.active_positions.append(pos)
        self.next_ticket += 1
        return pos
    
    def check_ticks(self, bid, ask, t_msc):
        remaining = []
        for pos in self.active_positions:
            resolved = False
            if pos["direction"] == 1: # BUY
                if bid <= pos["sl"]: 
                    self._close(pos, "LOSS", pos["sl"], t_msc)
                    resolved = True
                elif bid >= pos["tp"]: 
                    self._close(pos, "WIN", pos["tp"], t_msc)
                    resolved = True
            elif pos["direction"] == -1: # SELL
                if ask >= pos["sl"]: 
                    self._close(pos, "LOSS", pos["sl"], t_msc)
                    resolved = True
                elif ask <= pos["tp"]: 
                    self._close(pos, "WIN", pos["tp"], t_msc)
                    resolved = True
            
            if not resolved:
                remaining.append(pos)
                
        self.active_positions = remaining
        
    def force_close_all(self, bid, ask, t_msc):
        for pos in self.active_positions:
            close_price = bid if pos["direction"] == 1 else ask
            self._close(pos, "FORCE_CLOSE", close_price, t_msc)
        self.active_positions = []
        
    def _close(self, pos, outcome, close_price, exit_time):
        pnl = (close_price - pos["entry"]) * pos["direction"]
        pnl_r = pnl / pos["brick_size"]
        record = {
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
        return record


class SimEngine:
    def __init__(self, tick_provider: TickReplayProvider):
        self.provider = tick_provider
        self.broker = SimTrade_V3_Broker()
        self.optimizer = PathOptimizer()
        self.daily_summaries = []
        
        self.spread_threshold = 0.784
        self.ofi_threshold = 1.074
        
        self.log_count = 0
        self.MAX_LOGS = 100
        self.signals_log = []
    
    def run(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        days = self.provider.days
        
        logger.info(f"\n{'='*60}")
        logger.info(f" RULE SIMULATION START: {len(days)} trading days (Concurrent Trading ON)")
        logger.info(f"{'='*60}\n")
        
        for day_idx, day in enumerate(days):
            self._simulate_day(day, day_idx)
            
        self._generate_reports()
    
    def _simulate_day(self, day, day_idx):
        day_ticks = self.provider.get_day_ticks(day)
        if len(day_ticks) < 100: return
        
        day_open = day_ticks[0]['bid']
        brick_size = day_open * K_MULTIPLIER
        
        lookback_ticks = self.provider.get_lookback_ticks(day, lookback_days=7)
        all_history = lookback_ticks
        
        if len(lookback_ticks) > 1000:
            best_price, best_idx, _ = self.optimizer.find_optimal_anchor(all_history, brick_size)
            if best_price is None: best_price = day_open; best_idx = 0
        else:
            best_price = day_open; best_idx = 0
        
        renko = RenkoBuilder(best_price)
        renko.update_brick_size(brick_size, new_day_open=best_price)
        feature_engine = LiveFeatureEngine()
        feature_engine.update_brick_size(brick_size)
        
        warmup_bricks = 0
        for i, tick in enumerate(all_history):
            feature_engine.compute_vector(tick['bid'], tick['ask'], 0.0, 0.0, tick['time_msc'])
            if i >= best_idx:
                new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
                for brick in new_bricks:
                    warmup_bricks += 1
                    feature_engine.on_new_brick(brick)
        
        ticks_zscored = len(feature_engine.zs_ofi.deque)
        if not (warmup_bricks >= 10 and ticks_zscored >= 1000):
            self.daily_summaries.append({
                "day": str(day), "trades": 0, "wins": 0, "losses": 0, "pnl_r": 0.0, "warmup": False
            })
            return
        
        day_trades_before = len(self.broker.trade_log)
        ofi_peak = 0.0
        
        for tick in day_ticks:
            bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']
            
            feat_vec = feature_engine.compute_vector(bid, ask, 0.0, 0.0, t_msc)
            current_ofi = feat_vec[0] if feat_vec is not None else 0.0
            
            if abs(current_ofi) > abs(ofi_peak):
                ofi_peak = current_ofi
                
            if len(self.broker.active_positions) > 0:
                self.broker.check_ticks(bid, ask, t_msc)
            
            new_bricks = renko.update_tick(bid, t_msc)
            
            for brick in new_bricks:
                feature_engine.on_new_brick(brick)
                
                # Rule Evaluation matches statistical study definitions
                spread_current = ask - bid
                abs_ofi = abs(ofi_peak)
                
                action = 1 if (spread_current >= self.spread_threshold and abs_ofi <= self.ofi_threshold) else 0
                
                # Validation Logging
                if self.log_count < self.MAX_LOGS:
                    self.signals_log.append({
                        "timestamp": t_msc,
                        "spread_current": spread_current,
                        "abs_ofi_peak": abs_ofi,
                        "action": action
                    })
                    self.log_count += 1
                
                if action == 1:
                    is_buy = (brick.uptrend == 1)
                    bs = brick.brick_size
                    
                    if is_buy:
                        price, sl, tp, direction = bid, bid + bs, bid - bs, -1
                    else:
                        price, sl, tp, direction = ask, ask - bs, ask + bs, 1
                    
                    self.broker.execute(direction, price, sl, tp, bs, brick.timestamp)
                
                ofi_peak = 0.0
        
        if len(self.broker.active_positions) > 0 and len(day_ticks) > 0:
            self.broker.force_close_all(day_ticks[-1]['bid'], day_ticks[-1]['ask'], day_ticks[-1]['time_msc'])
        
        day_trades = self.broker.trade_log[day_trades_before:]
        wins = sum(1 for t in day_trades if t['outcome'] == 'WIN')
        losses = sum(1 for t in day_trades if t['outcome'] == 'LOSS')
        pnl_r = sum(t['pnl_r'] for t in day_trades)
        
        self.daily_summaries.append({
            "day": str(day), "trades": len(day_trades), "wins": wins, 
            "losses": losses, "pnl_r": round(pnl_r, 2), "warmup": True
        })
        
        wr = (wins / len(day_trades) * 100) if len(day_trades) > 0 else 0
        logger.info(f"  Day {day}: {len(day_trades)} trades, {wins}W/{losses}L, WR={wr:.0f}%, PnL={pnl_r:+.2f}R")
    
    def _generate_reports(self):
        trades = self.broker.trade_log
        resolved = [t for t in trades if t['outcome'] in ('WIN', 'LOSS')]
        total = len(resolved)
        wins = sum(1 for t in resolved if t['outcome'] == 'WIN')
        losses = total - wins
        
        wr = (wins / total * 100) if total > 0 else 0
        total_pnl_r = sum(t['pnl_r'] for t in trades)
        gross_profit = sum(t['pnl_r'] for t in trades if t['pnl_r'] > 0)
        gross_loss = abs(sum(t['pnl_r'] for t in trades if t['pnl_r'] < 0))
        pf = gross_profit / (gross_loss + 1e-8)
        
        logger.info(f"\n{'='*60}")
        logger.info(f" 2026 CONCURRENT OUT OF SAMPLE RULE TEST")
        logger.info(f" Total Trades: {len(trades)} (Resolved: {total})")
        logger.info(f" Wins: {wins} | Losses: {losses}")
        logger.info(f" Win Rate: {wr:.2f}%")
        logger.info(f" Profit Factor: {pf:.2f}")
        logger.info(f" Total PnL: {total_pnl_r:+.2f}R")
        logger.info(f"{'='*60}")
        
        pd.DataFrame(self.signals_log).to_csv(OUTPUT_DIR / "first_100_signals.csv", index=False)
        logger.info("Saved first 100 signals to first_100_signals.csv for validation.")
        
        cumulative_pnl = np.cumsum([t['pnl_r'] for t in trades])
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_pnl, linewidth=1.5, color='green')
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel("Trade Number")
        plt.ylabel("Cumulative PnL (R)")
        plt.title(f"2026 Rule Engine Equity Curve (WR={wr:.1f}%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "rule_equity_curve_2026.png", dpi=150)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=str, default=str(BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"))
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    provider = TickReplayProvider(args.ticks)
    engine = SimEngine(provider)
    engine.run()

if __name__ == "__main__":
    main()
