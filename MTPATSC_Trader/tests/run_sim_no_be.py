"""
Offline Simulator for MTPATSC Trader (NO BREAK-EVEN)
=====================================================
Identical to run_sim.py but with ALL break-even logic removed.
Trades can only resolve as WIN or LOSS — no BE outcomes possible.
This gives the raw, unmanipulated win rate and expectancy.
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add trader to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from bridge.renko import RenkoBuilder, K_MULTIPLIER
from bridge.mtpatsc_feature_engine import MTPatscFeatureEngine
from bridge.mtpatsc_predictor import MTPatscPredictor
from bridge.state import StateManager
from bridge.risk import RiskManager
from bridge.trade_logger import TradeLogger
from bridge.path_optimizer import PathOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def run_simulation(parquet_path: str):
    logger.info(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    if 'timestamp' in df.columns and 'time_msc' not in df.columns:
        df['time_msc'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
    
    df = df.sort_values('time_msc').reset_index(drop=True)
    
    bids = np.ascontiguousarray(df['bid'].values, dtype=np.float64)
    asks = np.ascontiguousarray(df['ask'].values, dtype=np.float64)
    times = np.ascontiguousarray(df['time_msc'].values, dtype=np.int64)
    
    logger.info(f"Loaded {len(bids)} ticks. Date range: {pd.to_datetime(times[0]*1000000)} to {pd.to_datetime(times[-1]*1000000)}")

    # Initialize components
    optimizer = PathOptimizer()
    scaler_path = str(BASE_DIR / "models" / "scalar_scaler.pkl")
    
    predictor = MTPatscPredictor(models_dir=str(BASE_DIR / "models"))
    try:
        predictor.load()
    except SystemExit:
        logger.error("Failed to load predictor. Exiting.")
        return
        
    state = StateManager(filepath=str(BASE_DIR / "logs" / "sim_no_be_state.json"))
    state.reset()
    risk = RiskManager()
    
    # Clear old sim trades CSV to prevent stale data from prior runs
    sim_csv = str(BASE_DIR / "logs" / "sim_no_be_trades.csv")
    if os.path.exists(sim_csv):
        os.remove(sim_csv)
    
    logger_obj = TradeLogger(filepath=sim_csv)
    
    # Simulation state variables
    active_ticket = 0
    ticket_counter = 1
    total_pnl = 0.0
    
    current_day = None
    renko = None
    feature_engine = None
    
    history_ticks = []  # Maintain recent history for path optimization during rollovers
    
    logger.info("Starting simulation loop (NO BREAK-EVEN)...")
    start_time = time.time()
    
    for i in range(len(bids)):
        bid = float(bids[i])
        ask = float(asks[i])
        time_msc = int(times[i])
        
        # Keep a rolling history of the last 50,000 ticks for path optimization
        history_ticks.append({'bid': bid, 'ask': ask, 'time_msc': time_msc})
        if len(history_ticks) > 50000:
            history_ticks.pop(0)
        
        # Check for daily rollover
        tick_date = datetime.fromtimestamp(time_msc / 1000.0, tz=timezone.utc).date()
        
        if current_day != tick_date:
            logger.info(f"--- ROLLOVER: {tick_date} ---")
            current_day = tick_date
            day_open = bid
            brick_size = day_open * K_MULTIPLIER
            state.update('daily_pnl', 0.0)
            
            # Replicate _warmup / _on_day_open logic
            if renko is None or len(history_ticks) < 1000:
                # First day or not enough history: static initialization
                renko = RenkoBuilder(day_open)
                renko.update_brick_size(brick_size, new_day_open=day_open)
                feature_engine = MTPatscFeatureEngine(scaler_path=scaler_path)
            else:
                # Subsequent days: Path Optimization
                logger.info("Running path optimization for new day...")
                best_price, best_idx, best_profit = optimizer.find_optimal_anchor(
                    history_ticks, brick_size
                )
                
                if best_price is not None:
                    renko = RenkoBuilder(best_price)
                    renko.update_brick_size(brick_size, new_day_open=best_price)
                    feature_engine = MTPatscFeatureEngine(scaler_path=scaler_path)
                    
                    # Replay history to build state
                    for h_idx in range(best_idx, len(history_ticks)):
                        t = history_ticks[h_idx]
                        new_bricks = renko.update_tick(t['bid'], t['time_msc'], ask=t['ask'])
                        for brick in new_bricks:
                            feature_engine.on_brick_close(brick)
                    logger.info(f"Path optimization complete. Replayed bricks: {renko.brick_count}")
                else:
                    renko.update_brick_size(brick_size, new_day_open=day_open)
        
        # 1. Update Renko
        new_bricks = renko.update_tick(bid, time_msc, ask=ask)
        
        for brick in new_bricks:
            # 2. Compute features
            tensors = feature_engine.on_brick_close(brick)
            
            # Note: feature_engine returns None if it doesn't have 5 bricks of history yet (warmup phase)
            if tensors is not None and active_ticket == 0:
                # 3. Predict
                result = predictor.predict(tensors, brick.uptrend)
                
                logger_obj.log_signal(brick.timestamp, brick.uptrend, result)
                
                if result['action'] == 1:
                    spread = ask - bid
                    # Risk checks
                    daily_pnl = state.get('daily_pnl', 0.0)
                    if not risk.check_daily_limit(daily_pnl, renko.brick_size):
                        continue
                        
                    if not risk.check_spread(spread, renko.brick_size):
                        continue
                        
                    setup_type = result['setup_type']
                    trade_direction = result['direction']
                    rr = result['rr']
                    bs = renko.brick_size
                    close_price = brick.close
                    
                    if setup_type == 1:
                        if trade_direction == 1:
                            entry = close_price + spread
                            sl = entry - bs
                            tp = entry + bs
                        else:
                            entry = close_price - spread
                            sl = entry + bs
                            tp = entry - bs
                    elif setup_type == 3:
                        if trade_direction == 1:
                            entry = ask
                            tp = entry + 2.0 * bs
                            sl = entry - bs
                        else:
                            entry = bid
                            tp = entry - 2.0 * bs
                            sl = entry + bs
                    elif setup_type == 4:
                        if trade_direction == 1:
                            entry = ask
                            tp = entry + 3.0 * bs
                            sl = entry - bs
                        else:
                            entry = bid
                            tp = entry - 3.0 * bs
                            sl = entry + bs
                    else:
                        continue
                        
                    # Execute
                    active_ticket = ticket_counter
                    ticket_counter += 1
                    
                    state.update("active_ticket", active_ticket)
                    state.update("active_direction", trade_direction)
                    state.update("active_entry", entry)
                    state.update("active_sl", sl)
                    state.update("active_tp", tp)
                    state.update("active_brick_size", bs)
                    state.update("active_setup_type", f"T{setup_type}")
                    state.update("active_rr", rr)
                    
                    logger_obj.log_order(active_ticket, entry, sl, tp, trade_direction, entry_spread_pts=spread)

        # 4. Manage open position — NO BREAK-EVEN, only TP/SL
        if active_ticket != 0:
            direction = state.get("active_direction")
            sl = state.get("active_sl")
            tp = state.get("active_tp")
            entry = state.get("active_entry")
                    
            # Check TP / SL — pure WIN or LOSS only
            outcome = None
            exit_price = 0.0
            
            if direction == 1:  # BUY
                if bid <= sl:
                    outcome = "LOSS"
                    exit_price = sl
                elif bid >= tp:
                    outcome = "WIN"
                    exit_price = tp
            else:  # SELL
                if ask >= sl:
                    outcome = "LOSS"
                    exit_price = sl
                elif ask <= tp:
                    outcome = "WIN"
                    exit_price = tp
                    
            if outcome:
                # Calculate PnL in points
                if direction == 1:
                    pnl = exit_price - entry
                else:
                    pnl = entry - exit_price
                    
                total_pnl += pnl
                
                # Update daily PnL
                state.update('daily_pnl', state.get('daily_pnl', 0.0) + pnl)
                
                logger_obj.log_outcome(active_ticket, outcome, pnl)
                
                # Reset state
                active_ticket = 0
                state.update("active_ticket", 0)

    # Finish
    elapsed = time.time() - start_time
    logger.info(f"Simulation finished in {elapsed:.2f}s. Processed {len(bids)} ticks.")
    
    report_path = str(BASE_DIR / "logs" / "sim_no_be_session_report.md")
    logger_obj.generate_session_report(report_path=report_path)
    
    with open(report_path, "r") as f:
        print("\n" + f.read())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parquet_path = sys.argv[1]
    else:
        parquet_path = "/Users/gopo/Quant Projects/CAPSTONE/Overshot/Data/Raw/2015.parquet"
    run_simulation(parquet_path)
