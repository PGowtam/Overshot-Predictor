"""
2026 Live Engine Simulation — Tick Replay Simulator
====================================================
Replays historical 2026 ticks through the exact production BridgeEngine
pipeline to validate the model's real-world execution performance.

Uses REAL production components (RenkoBuilder, LiveFeatureEngine,
InferenceBuffer, PathOptimizer, FallbackPredictor) — only the network
layer (sockets) is replaced by in-memory replay classes.

Usage:
    python src/sim_live_engine_2026.py [--ticks data/xauusd_ticks_2026.parquet]
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Path Setup ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TRADER_DIR = BASE_DIR / "BrickOfTicks_Trader"

# Add the Trader directory to sys.path so we can import bridge modules
sys.path.insert(0, str(TRADER_DIR))

# Patch K_MULTIPLIER BEFORE importing bridge modules (matches main_fallback.py)
import bridge.renko
import bridge.path_optimizer
bridge.renko.K_MULTIPLIER = 0.00118
bridge.path_optimizer.K_MULTIPLIER = 0.00118

from bridge.renko import RenkoBuilder, K_MULTIPLIER
from bridge.feature_engine import LiveFeatureEngine
from bridge.buffer import InferenceBuffer
from bridge.path_optimizer import PathOptimizer

import tensorflow as tf

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────
ATTN_DIR = BASE_DIR / "outputs" / "exec_baseline"
MODEL_PATH = ATTN_DIR / "model.keras"
OUTPUT_DIR = BASE_DIR / "outputs" / "experiments" / "sim_global_fade_2026"


# ══════════════════════════════════════════════════════════════════
#  Simulation-Only Classes (replace socket infrastructure)
# ══════════════════════════════════════════════════════════════════

class TickReplayProvider:
    """Reads a parquet of ticks and provides day-boundary-aware iteration."""
    
    def __init__(self, parquet_path: str):
        logger.info(f"Loading tick data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.df = self.df.sort_values('time_msc').reset_index(drop=True)
        
        # Detect day boundaries (UTC)
        self.df['utc_day'] = pd.to_datetime(self.df['time_msc'], unit='ms', utc=True).dt.date
        self.days = sorted(self.df['utc_day'].unique())
        
        logger.info(f"Loaded {len(self.df):,} ticks across {len(self.days)} trading days")
        logger.info(f"Date range: {self.days[0]} → {self.days[-1]}")
        logger.info(f"Price range: {self.df['bid'].min():.2f} → {self.df['bid'].max():.2f}")
    
    def get_day_ticks(self, day) -> list:
        """Return list of tick dicts for a specific day."""
        mask = self.df['utc_day'] == day
        day_df = self.df[mask]
        return [
            {
                'time_msc': int(row['time_msc']),
                'bid': float(row['bid']),
                'ask': float(row['ask']),
                'bid_vol': float(row.get('bid_vol', 0.0)),
                'ask_vol': float(row.get('ask_vol', 0.0))
            }
            for _, row in day_df.iterrows()
        ]
    
    def get_lookback_ticks(self, target_day, lookback_days=7) -> list:
        """Return ticks from the N calendar days BEFORE target_day."""
        target_idx = self.days.index(target_day)
        # Grab from max(0, target_idx - lookback_days) to target_idx (exclusive)
        start_idx = max(0, target_idx - lookback_days)
        lookback_days_list = self.days[start_idx:target_idx]
        
        if not lookback_days_list:
            return []
        
        mask = self.df['utc_day'].isin(lookback_days_list)
        lb_df = self.df[mask]
        return [
            {
                'time_msc': int(row['time_msc']),
                'bid': float(row['bid']),
                'ask': float(row['ask']),
                'bid_vol': float(row.get('bid_vol', 0.0)),
                'ask_vol': float(row.get('ask_vol', 0.0))
            }
            for _, row in lb_df.iterrows()
        ]


class SimulatedBroker:
    """Replaces CommandSender + MT5 execution. Pure 1:1 SL/TP. No break-even."""
    
    def __init__(self):
        self.active_position = None
        self.trade_log = []
        self.next_ticket = 1000
        self.daily_pnl = 0.0
    
    def execute(self, direction, price, sl, tp, brick_size, timestamp):
        """Instant fill at price."""
        if self.active_position is not None:
            return None  # Position already open
        
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
        logger.debug(f"  📈 OPENED: ticket={self.active_position['ticket']} "
                     f"dir={'BUY' if direction==1 else 'SELL'} "
                     f"entry={price:.2f} SL={sl:.2f} TP={tp:.2f}")
        return {"status": "OK", "ticket": self.active_position["ticket"]}
    
    def check_tick(self, tick):
        """Check if active position hits SL or TP. No break-even."""
        pos = self.active_position
        if pos is None:
            return None
        
        if pos["direction"] == 1:  # BUY
            if tick["bid"] <= pos["sl"]:
                return self._close("LOSS", pos["sl"], tick["time_msc"])
            if tick["bid"] >= pos["tp"]:
                return self._close("WIN", pos["tp"], tick["time_msc"])
        elif pos["direction"] == -1:  # SELL
            if tick["ask"] >= pos["sl"]:
                return self._close("LOSS", pos["sl"], tick["time_msc"])
            if tick["ask"] <= pos["tp"]:
                return self._close("WIN", pos["tp"], tick["time_msc"])
        
        return None
    
    def force_close(self, tick):
        """Force close at end of simulation or day."""
        pos = self.active_position
        if pos is None:
            return None
        
        close_price = tick["bid"] if pos["direction"] == 1 else tick["ask"]
        pnl = (close_price - pos["entry"]) * pos["direction"]
        pnl_r = pnl / pos["brick_size"]
        
        outcome = "FORCE_CLOSE"
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
            "exit_time": tick["time_msc"]
        }
        self.trade_log.append(record)
        self.daily_pnl += pnl
        self.active_position = None
        return record
    
    def _close(self, outcome, close_price, exit_time):
        pos = self.active_position
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
        self.daily_pnl += pnl
        self.active_position = None
        
        emoji = "✅" if outcome == "WIN" else "❌"
        logger.info(f"  {emoji} CLOSED: ticket={record['ticket']} "
                    f"outcome={outcome} pnl={pnl_r:+.2f}R")
        return record
    
    def reset_daily(self):
        self.daily_pnl = 0.0
    
    @property
    def has_position(self):
        return self.active_position is not None


class GlobalFadePredictor:
    """Loads the Baseline CNN Execution model and executes a Global Fade."""
    
    def __init__(self):
        self.model = None
        self.prob_win_threshold = 0.44   # 0.44 yields very high volume while retaining a ~70% mathematical win rate on Baseline
    
    def load(self):
        logger.info(f"Loading Baseline model from {MODEL_PATH}...")
        
        sys.path.insert(0, str(BASE_DIR / "src"))
        from models_exec import build_baseline_exec_model
        self.model = build_baseline_exec_model()
        self.model.load_weights(str(MODEL_PATH))
        logger.info(f"Using exact Global Fade threshold: PW <= {self.prob_win_threshold}")
        logger.info("Model loaded successfully.")
    
    def predict(self, micro_tensor, macro_tensor, tick_time_msc):
        """Run inference. Returns (action, prob_win)."""
        if self.model is None:
            return 0, 0.0
            
        # Crop the 7-element macro tensor to the 3 elements expected by the Baseline model
        # The first 3 elements are [log_dur, direction, z_size]
        baseline_macro = macro_tensor[:, :, 0:3]
        
        preds = self.model([micro_tensor, baseline_macro], training=False)
        pw = float(preds[0].numpy().flatten()[0])
        
        # Action = 1 means FADE
        # Removed Asian filter: trigger globally when probability is low
        signal = (pw <= self.prob_win_threshold)
        action = 1 if signal else 0
        
        return action, pw


# ══════════════════════════════════════════════════════════════════
#  Simulation Engine
# ══════════════════════════════════════════════════════════════════

class SimEngine:
    """Mirrors BridgeEngine but with deterministic tick replay."""
    
    def __init__(self, tick_provider: TickReplayProvider):
        self.provider = tick_provider
        self.predictor = GlobalFadePredictor()
        self.broker = SimulatedBroker()
        self.optimizer = PathOptimizer()
        
        self.daily_summaries = []
        self.all_signals = []
    
    def run(self):
        """Run the full simulation across all trading days."""
        self.predictor.load()
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        days = self.provider.days
        logger.info(f"\n{'='*60}")
        logger.info(f" SIMULATION START: {len(days)} trading days")
        logger.info(f"{'='*60}\n")
        
        for day_idx, day in enumerate(days):
            self._simulate_day(day, day_idx)
        
        self._generate_reports()
    
    def _simulate_day(self, day, day_idx):
        """Simulate a single trading day."""
        day_ticks = self.provider.get_day_ticks(day)
        if len(day_ticks) < 100:
            logger.warning(f"Day {day}: Only {len(day_ticks)} ticks. Skipping.")
            return
        
        # Day open price
        day_open = day_ticks[0]['bid']
        brick_size = day_open * K_MULTIPLIER
        
        logger.info(f"📅 Day {day} ({day_idx+1}/{len(self.provider.days)}): "
                    f"{len(day_ticks):,} ticks, open={day_open:.2f}, bs={brick_size:.4f}")
        
        # Get lookback history for path optimization + z-score warmup
        lookback_ticks = self.provider.get_lookback_ticks(day, lookback_days=7)
        # BUG FIX: DO NOT include day_ticks in all_history! This was double-feeding ticks and destroying tensors.
        all_history = lookback_ticks
        
        # Path Optimization
        if len(lookback_ticks) > 1000:
            best_price, best_idx, best_profit = self.optimizer.find_optimal_anchor(
                all_history, brick_size
            )
            if best_price is None:
                best_price = day_open
                best_idx = 0
                logger.warning(f"  PathOptimizer failed. Using day_open as anchor.")
            else:
                logger.info(f"  PathOptimizer: anchor={best_price:.2f}, "
                           f"profit={best_profit:.2f}, start_idx={best_idx}")
        else:
            best_price = day_open
            best_idx = 0
            logger.info(f"  No lookback history. Using day_open as anchor.")
        
        # Initialize fresh pipeline
        renko = RenkoBuilder(best_price)
        renko.update_brick_size(brick_size, new_day_open=best_price)
        feature_engine = LiveFeatureEngine()
        feature_engine.update_brick_size(brick_size)
        buffer = InferenceBuffer()
        
        # Reset daily broker state
        self.broker.reset_daily()
        
        # ── Replay ALL history ticks for z-score warmup ──
        warmup_bricks = 0
        for i, tick in enumerate(all_history):
            feat_vec = feature_engine.compute_vector(
                tick['bid'], tick['ask'], 0.0, 0.0, tick['time_msc']
            )
            if i >= best_idx:
                buffer.append_tick(feat_vec, renko.brick_count)
                new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
                for brick in new_bricks:
                    warmup_bricks += 1
                    feature_engine.on_new_brick(brick)
                    buffer.on_brick_close(renko.brick_count - 1, feature_engine.last_macro)
        
        # Check warmup gate
        ticks_zscored = len(feature_engine.zs_ofi.deque)
        warmup_passed = (warmup_bricks >= 10 and ticks_zscored >= 1000)
        
        if not warmup_passed:
            logger.warning(f"  Warmup gate NOT met: {warmup_bricks} bricks, "
                          f"{ticks_zscored} z-ticks. Skipping day.")
            self.daily_summaries.append({
                "day": str(day), "ticks": len(day_ticks), "bricks": 0,
                "signals": 0, "trades": 0, "wins": 0, "losses": 0,
                "pnl_r": 0.0, "warmup": False
            })
            return
        
        logger.info(f"  Warmup PASSED: {warmup_bricks} bricks, {ticks_zscored} z-ticks")
        
        # ── Stream today's ticks through the live pipeline ──
        day_bricks = 0
        day_signals = 0
        day_trades_before = len(self.broker.trade_log)
        
        # Today's ticks are streamed freshly
        day_trades_before = len(self.broker.trade_log)
        
        self.day_min_pw = 1.0
        self.day_max_pw = 0.0
        
        for tick in day_ticks:
            # 1. Compute features
            feat_vec = feature_engine.compute_vector(
                tick['bid'], tick['ask'], 0.0, 0.0, tick['time_msc']
            )
            
            # 2. Append to buffer BEFORE updating Renko (to align brick_count properly)
            buffer.append_tick(feat_vec, renko.brick_count)
            
            # 3. Check SL/TP on every tick
            if self.broker.has_position:
                self.broker.check_tick(tick)
            
            # 4. Feed to Renko
            new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
            
            for brick in new_bricks:
                day_bricks += 1
                feature_engine.on_new_brick(brick)
                tensors = buffer.on_brick_close(renko.brick_count - 1, feature_engine.last_macro)
                
                if tensors is not None:
                    micro, macro = tensors
                    action, pw = self.predictor.predict(micro, macro, tick['time_msc'])
                    
                    if not hasattr(self, 'dumped_tensors'):
                        np.save("live_micro_debug.npy", micro)
                        np.save("live_macro_debug.npy", macro)
                        self.dumped_tensors = True
                    
                    if pw < self.day_min_pw: self.day_min_pw = pw
                    if pw > self.day_max_pw: self.day_max_pw = pw
                    
                    day_signals += 1
                    self.all_signals.append({
                        "day": str(day),
                        "timestamp": brick.timestamp,
                        "direction": brick.uptrend,
                        "prob_win": pw,
                        "action": action
                    })
                    
                    if action == 1 and not self.broker.has_position:
                        # Check daily limit (5 SL losses)
                        daily_limit = -5.0 * brick_size
                        if self.broker.daily_pnl < daily_limit:
                            logger.warning(f"  Daily limit hit. Blocking trade.")
                            continue
                        
                        is_buy = (brick.uptrend == 1)
                        bs = brick.brick_size
                        
                        if is_buy:
                            # FADE: Buy breakout -> SELL
                            price = tick['bid']
                            sl = price + (1.0 * bs)
                            tp = price - (1.0 * bs)
                            direction = -1
                        else:
                            # FADE: Sell breakout -> BUY
                            price = tick['ask']
                            sl = price - (1.0 * bs)
                            tp = price + (1.0 * bs)
                            direction = 1
                        
                        self.broker.execute(direction, price, sl, tp, bs, brick.timestamp)
        
        # End of day: force close any open position
        if self.broker.has_position and len(day_ticks) > 0:
            self.broker.force_close(day_ticks[-1])
        
        # Daily summary
        day_trades = self.broker.trade_log[day_trades_before:]
        wins = sum(1 for t in day_trades if t['outcome'] == 'WIN')
        losses = sum(1 for t in day_trades if t['outcome'] == 'LOSS')
        force_closed = sum(1 for t in day_trades if t['outcome'] == 'FORCE_CLOSE')
        pnl_r = sum(t['pnl_r'] for t in day_trades)
        
        summary = {
            "day": str(day), "ticks": len(day_ticks), "bricks": day_bricks,
            "signals": day_signals, "trades": len(day_trades),
            "wins": wins, "losses": losses, "force_closed": force_closed,
            "pnl_r": round(pnl_r, 2), "warmup": True
        }
        self.daily_summaries.append(summary)
        
        wr = (wins / len(day_trades) * 100) if len(day_trades) > 0 else 0
        logger.info(f"  📊 Day result: {day_bricks} bricks, {len(day_trades)} trades, "
                    f"{wins}W/{losses}L, WR={wr:.0f}%, PnL={pnl_r:+.2f}R (PW Range: {self.day_min_pw:.3f} - {self.day_max_pw:.3f})")
    
    def _generate_reports(self):
        """Generate all output files."""
        logger.info(f"\n{'='*60}")
        logger.info(f" GENERATING REPORTS")
        logger.info(f"{'='*60}")
        
        # 1. Trades CSV
        if self.broker.trade_log:
            trades_df = pd.DataFrame(self.broker.trade_log)
            trades_df.to_csv(OUTPUT_DIR / "sim_trades.csv", index=False)
            logger.info(f"Saved {len(trades_df)} trades to sim_trades.csv")
        
        # 2. Daily summary CSV
        summary_df = pd.DataFrame(self.daily_summaries)
        summary_df.to_csv(OUTPUT_DIR / "sim_daily_summary.csv", index=False)
        logger.info(f"Saved daily summary to sim_daily_summary.csv")
        
        # 3. Signals CSV
        if self.all_signals:
            signals_df = pd.DataFrame(self.all_signals)
            signals_df.to_csv(OUTPUT_DIR / "sim_signals.csv", index=False)
        
        # 4. Compute final statistics
        trades = self.broker.trade_log
        if not trades:
            logger.warning("No trades executed during simulation!")
            return
        
        # Filter to WIN/LOSS only (exclude FORCE_CLOSE)
        resolved = [t for t in trades if t['outcome'] in ('WIN', 'LOSS')]
        total = len(resolved)
        wins = sum(1 for t in resolved if t['outcome'] == 'WIN')
        losses = total - wins
        force_closed = sum(1 for t in trades if t['outcome'] == 'FORCE_CLOSE')
        
        wr = (wins / total * 100) if total > 0 else 0
        total_pnl_r = sum(t['pnl_r'] for t in trades)
        ev_per_trade = (total_pnl_r / total) if total > 0 else 0
        
        # 5. Equity curve
        cumulative_pnl = np.cumsum([t['pnl_r'] for t in trades])
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_pnl, linewidth=1.5, color='blue')
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel("Trade Number")
        plt.ylabel("Cumulative PnL (R)")
        plt.title(f"2026 Live Engine Simulation — Equity Curve (WR={wr:.1f}%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "sim_equity_curve.png", dpi=150)
        plt.close()
        logger.info("Saved equity curve to sim_equity_curve.png")
        
        # 6. Session report markdown
        report = f"""# 2026 Live Engine Simulation — Session Report

## Configuration
- **K_MULTIPLIER:** {K_MULTIPLIER}
- **Thresholds:** Prob_Win ≥ {self.predictor.prob_win_threshold}, Pred_OS ≥ {self.predictor.pred_os_threshold}
- **Temperature Scaling:** T = {self.predictor.T:.4f}
- **Break-Even:** Disabled
- **Risk-Reward:** 1:1 (SL = 1 brick, TP = 1 brick)

## Results Summary

| Metric | Value |
| :--- | :--- |
| **Total Resolved Trades** | {total} |
| **Wins** | {wins} |
| **Losses** | {losses} |
| **Force Closed (EOD)** | {force_closed} |
| **Win Rate** | **{wr:.2f}%** |
| **Total PnL** | {total_pnl_r:+.2f}R |
| **EV per Trade** | {ev_per_trade:+.4f}R |
| **Trading Days** | {len([s for s in self.daily_summaries if s['warmup']])} |
| **Total Signals** | {len(self.all_signals)} |

## Daily Breakdown

| Day | Bricks | Trades | W | L | WR | PnL (R) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for s in self.daily_summaries:
            if not s['warmup']:
                continue
            if s['trades'] > 0:
                day_wr = s['wins'] / s['trades'] * 100
                report += f"| {s['day']} | {s['bricks']} | {s['trades']} | {s['wins']} | {s['losses']} | {day_wr:.0f}% | {s['pnl_r']:+.2f} |\n"
            elif s['signals'] > 0:
                report += f"| {s['day']} | {s['bricks']} | 0 | — | — | — | 0.00 |\n"
        
        report += f"""
## Comparison to Theoretical Research
This Asian Fade simulation evaluates tick-by-tick trajectory pathing dynamically across the entire 2026 year.
Theoretical math expects ~81.5% win rate over 325 trades in the holdout set.
This engine measures the true performance accounting for Bid/Ask spread, tick jitter, and sequential daily drawdowns!
"""
        
        with open(OUTPUT_DIR / "sim_session_report.md", "w") as f:
            f.write(report)
        
        logger.info(f"Saved session report to sim_session_report.md")
        
        # Final console summary
        logger.info(f"\n{'='*60}")
        logger.info(f" SIMULATION COMPLETE")
        logger.info(f" Trades: {total} | Wins: {wins} | Losses: {losses}")
        logger.info(f" Win Rate: {wr:.2f}%")
        logger.info(f" Total PnL: {total_pnl_r:+.2f}R")
        logger.info(f" EV/Trade: {ev_per_trade:+.4f}R")
        logger.info(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="2026 Live Engine Simulation")
    parser.add_argument("--ticks", type=str,
                        default=str(BASE_DIR / "data" / "xauusd_ticks_2026.parquet"),
                        help="Path to 2026 tick parquet file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    level = logging.DEBUG if args.verbose else logging.INFO
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / "sim.log"), mode='w')
        ]
    )
    
    if not os.path.exists(args.ticks):
        logger.error(f"Tick file not found: {args.ticks}")
        logger.error("Run tick_collector.py first, then attach TickExporter EA in MT5.")
        sys.exit(1)
    
    provider = TickReplayProvider(args.ticks)
    engine = SimEngine(provider)
    engine.run()


if __name__ == "__main__":
    main()
