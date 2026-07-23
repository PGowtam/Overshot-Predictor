import os
import sys

# No TF in this process — predictions run in an isolated subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import logging
import subprocess
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ctypes

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Simulator2026")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MTPATSC_DIR = BASE_DIR / "MTPATSC"
OUTPUT_DIR = MTPATSC_DIR / "outputs" / "setup_classifier"
SIM_OUT_DIR = OUTPUT_DIR / "sim_2026"

class FeatureLabelRow(ctypes.Structure):
    _fields_ = [
        ("brick_id", ctypes.c_int),
        ("timestamp", ctypes.c_int64),
        ("direction", ctypes.c_int),
        ("close_price", ctypes.c_double),
        ("open_price", ctypes.c_double),
        ("brick_size", ctypes.c_double),
        
        ("t1_win", ctypes.c_int),
        ("t1_y_mag", ctypes.c_double),
        ("t2_win", ctypes.c_int),
        ("t2_y_mag", ctypes.c_double),
        ("t2_filled", ctypes.c_int),
        ("t3_win", ctypes.c_int),
        ("t3_y_mag", ctypes.c_double),
        ("t4_win", ctypes.c_int),
        ("t4_y_mag", ctypes.c_double),
        ("t4_filled", ctypes.c_int),
        ("label", ctypes.c_int),
        ("exclude_flag", ctypes.c_int),
        ("brick_duration_seconds", ctypes.c_int64),
        
        ("ancs_fine", ctypes.c_float * 60),
        ("ancs_coarse", ctypes.c_float * 30),
        ("candle_features", ctypes.c_float * 15),
        ("momentum", ctypes.c_float * 19),
        ("history", ctypes.c_float * 150)
    ]

def load_cpp_library():
    lib_path = MTPATSC_DIR / "src" / "libmtpatsc_engine.dylib"
    if not lib_path.exists():
        raise FileNotFoundError(f"C++ library not found at {lib_path}. Compile it first.")
        
    lib = ctypes.CDLL(str(lib_path))
    lib.generate_dataset.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int)
    ]
    lib.generate_dataset.restype = ctypes.POINTER(FeatureLabelRow)
    lib.free_dataset.argtypes = [ctypes.POINTER(FeatureLabelRow)]
    lib.free_dataset.restype = None
    return lib

def scan_exit_ticks(bids, asks, times, start_idx, direction, entry, tp, sl):
    """
    Scans forward tick-by-tick to find when TP or SL is touched.
    IMPORTANT: Starts from start_idx + 1 to skip the fill tick itself.
    This prevents instant ghost wins where the TP is already past at the
    exact moment of entry (a lookahead artifact from fast-moving bricks).
    Returns: (is_win, exit_time)
    """
    for i in range(start_idx + 1, len(bids)):
        bid = bids[i]
        ask = asks[i]
        t = times[i]
        
        if direction == 1: # Long
            if bid >= tp:
                return True, t
            if bid <= sl:
                return False, t
        else: # Short
            if ask <= tp:
                return True, t
            if ask >= sl:
                return False, t
                
    # If not resolved by end of file, default to loss at the last tick time
    return False, times[-1]

def main():
    k_multiplier = 0.00118
    SIM_OUT_DIR = OUTPUT_DIR / "sim_2026_5ers"
    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    ticks_path = BASE_DIR / "Data" / "xauusd_ticks_5ers_2026_clean.parquet"
    if not ticks_path.exists():
        ticks_path = BASE_DIR / "Data" / "xauusd_ticks_5ers_2026.parquet"
    if not ticks_path.exists():
        # Fallback to lower-case folder if exists
        ticks_path = BASE_DIR / "data" / "xauusd_ticks_5ers_2026.parquet"
        if not ticks_path.exists():
            logger.error(f"Ticks file not found at {ticks_path}")
            return
            
    logger.info(f"Loading 5ers 2026 tick data from {ticks_path}...")
    df_ticks = pd.read_parquet(ticks_path)
    if 'timestamp' in df_ticks.columns and 'time_msc' not in df_ticks.columns:
        df_ticks['time_msc'] = pd.to_datetime(df_ticks['timestamp']).astype('int64') // 10**6
    df_ticks = df_ticks.sort_values('time_msc').reset_index(drop=True)
    
    bids = np.ascontiguousarray(df_ticks['bid'].values, dtype=np.float64)
    asks = np.ascontiguousarray(df_ticks['ask'].values, dtype=np.float64)
    times = np.ascontiguousarray(df_ticks['time_msc'].values, dtype=np.int64)
    
    # Check if bricks are already generated to save time
    bricks_path = OUTPUT_DIR / "sim_2026_5ers_bricks.parquet"
    extractor_script = MTPATSC_DIR / "src" / "sc_sim_extractor.py"
    
    if bricks_path.exists():
        logger.info(f"Bricks parquet already exists ({bricks_path}), skipping C++ extraction.")
    else:
        logger.info("Executing C++ Feature Engine in isolated subprocess...")
        # Since the extractor has hardcoded paths, we need to pass the custom paths as args
        subprocess.run([sys.executable, str(extractor_script), str(ticks_path), str(bricks_path)], check=True)
    
    # Load the generated bricks dataset
    logger.info(f"Loading generated bricks from {bricks_path}...")
    df_bricks = pd.read_parquet(bricks_path)
    
    bricks_data = []
    for _, row in df_bricks.iterrows():
        bricks_data.append({
            "brick_id": int(row["brick_id"]),
            "timestamp": int(row["timestamp"]),
            "direction": int(row["direction"]),
            "close_price": float(row["close_price"]),
            "open_price": float(row["open_price"]),
            "brick_size": float(row["brick_size"]),
            "t1_win": int(row.get("t1_win", 0)),
            "t2_win": int(row.get("t2_win", 0)),
            "t3_win": int(row.get("t3_win", 0)),
            "t4_win": int(row.get("t4_win", 0)),
            "label": int(row.get("label", 0)),
            "ancs_fine": np.stack(row["ancs_fine"]).astype(np.float32),
            "ancs_coarse": np.stack(row["ancs_coarse"]).astype(np.float32),
            "history": np.stack([np.stack(x) for x in row["history"]]).astype(np.float32),
            "candle_features": np.array(row["candle_features"], dtype=np.float32),
            "momentum": np.array(row["momentum"], dtype=np.float32)
        })
        
    logger.info(f"Loaded {len(bricks_data)} valid closed bricks for 5ers 2026.")
    if not bricks_data:
        logger.error("No valid bricks generated.")
        return
        
    # Run TF model in an isolated subprocess to avoid macOS Metal/OpenMP deadlock
    predictor_script = MTPATSC_DIR / "src" / "sc_sim_predictor.py"
    probs_path = OUTPUT_DIR / "sim_2026_5ers_probs.npy"
    if probs_path.exists():
        logger.info(f"Probabilities cache found, skipping prediction subprocess.")
    else:
        logger.info("Executing TensorFlow Predictor in isolated subprocess...")
        subprocess.run([sys.executable, str(predictor_script), str(bricks_path), str(probs_path)], check=True)
        
    logger.info(f"Loading probabilities from {probs_path}...")
    probs = np.load(str(probs_path))
    pred_classes = np.argmax(probs, axis=1)
    logger.info(f"Probabilities loaded: shape={probs.shape}")
    
    # Load calibration config
    config_path = OUTPUT_DIR / "config.json"
    if not config_path.exists():
        logger.error("config.json not found.")
        return
    with open(config_path) as f:
        config = json.load(f)
    veto_threshold = config.get("T0_veto_threshold", 0.40)
    
    # Calculate winrates at multiple thresholds
    logger.info("\n==================================================")
    logger.info(" WINRATES AT MULTIPLE THRESHOLDS (OOS 2026 DATA)")
    logger.info("==================================================")
    thresholds_to_test = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    win_flags_arrays = {
        1: np.array([b['t1_win'] for b in bricks_data]),
        2: np.array([b['t2_win'] for b in bricks_data]),
        3: np.array([b['t3_win'] for b in bricks_data]),
        4: np.array([b['t4_win'] for b in bricks_data])
    }
    
    thresholds_table_md = "### Winrate of setups at multiple thresholds\n\n"
    for setup_class in [1, 2, 3, 4]:
        logger.info(f"\nSetup T{setup_class} Performance:")
        logger.info(f"  {'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10}")
        logger.info("  " + "-" * 34)
        thresholds_table_md += f"#### Setup T{setup_class}\n\n| Threshold | Trades | Win Rate |\n| :--- | :--- | :--- |\n"
        t_win = win_flags_arrays[setup_class]
        for th in thresholds_to_test:
            # Veto if P(T0) > veto_threshold
            mask = (pred_classes == setup_class) & (probs[:, setup_class] >= th) & (probs[:, 0] <= veto_threshold)
            n_tr = int(mask.sum())
            if n_tr > 0:
                wr = float(np.mean(t_win[mask]))
                logger.info(f"  {th:<10.2f} | {n_tr:<8} | {wr:<10.2%}")
                thresholds_table_md += f"| {th:.2f} | {n_tr} | {wr:.2%} |\n"
            else:
                logger.info(f"  {th:<10.2f} | {0:<8} | {'N/A':<10}")
                thresholds_table_md += f"| {th:.2f} | 0 | N/A |\n"
        thresholds_table_md += "\n"
    logger.info("==================================================\n")
    
    # Trading Loop Variables
    active_trade = None # None or dict
    trade_log = []
    daily_summaries = {}
    last_signal_brick_time = -1  # Guard: prevent multiple trades on same-timestamp bricks
    
    T1_threshold = config.get("T1_threshold", 1.0)
    T2_threshold = config.get("T2_threshold", 1.0)
    T3_threshold = config.get("T3_threshold", 1.0)
    T4_threshold = config.get("T4_threshold", 1.0)
    veto_threshold = config.get("T0_veto_threshold", 0.40)
    
    thresholds = {1: T1_threshold, 2: T2_threshold, 3: T3_threshold, 4: T4_threshold}
    rr_profiles = {1: 1.0, 2: 2.0, 3: 2.0, 4: 3.0}
    
    logger.info("Starting live execution simulation loop...")
    for i in range(len(bricks_data)):
        brick = bricks_data[i]
        b_time = brick['timestamp']
        b_date = datetime.utcfromtimestamp(b_time / 1000.0).date()
        date_str = b_date.strftime("%Y-%m-%d")
        
        # Initialize day summary if new
        if date_str not in daily_summaries:
            daily_summaries[date_str] = {
                "date": date_str, "trades": 0, "wins": 0, "losses": 0,
                "daily_pnl_R": 0.0, "bricks_count": 0
            }
        daily_summaries[date_str]["bricks_count"] += 1
        
        # 1. Update active position check
        if active_trade is not None:
            # If the current brick timestamp is after the exit time of the active trade, close it
            if b_time >= active_trade['exit_time']:
                trade_log.append(active_trade)
                entry_date_str = datetime.utcfromtimestamp(active_trade['entry_time'] / 1000.0).date().strftime("%Y-%m-%d")
                daily_summaries[entry_date_str]["trades"] += 1
                if active_trade['pnl_R'] > 0.0:
                    daily_summaries[entry_date_str]["wins"] += 1
                else:
                    daily_summaries[entry_date_str]["losses"] += 1
                daily_summaries[entry_date_str]["daily_pnl_R"] += active_trade['pnl_R']
                active_trade = None
                
        # 2. Check daily drawdown stop
        # If the daily PnL has reached <= -5.0 R, we stand aside for this day
        if daily_summaries[date_str]["daily_pnl_R"] <= -5.0:
            continue
            
        # 3. Check Signal Generation
        if active_trade is None:
            # Guard: skip bricks that share the exact same millisecond timestamp as
            # the last signal. This prevents a cascade of ghost trades when multiple
            # consecutive bricks happen to close at the identical tick (e.g., during
            # a gap open or replay boundary).
            if b_time == last_signal_brick_time:
                continue
            
            p_t0 = probs[i, 0]
            if p_t0 > veto_threshold:
                continue # Vetoed
                
            pred_c = pred_classes[i]
            if pred_c not in thresholds:
                continue # Pred T0
                
            theta = thresholds[pred_c]
            if theta >= 1.0:
                continue # Setup type is disabled
                
            p_signal = probs[i, pred_c]
            if p_signal < theta:
                continue # Below threshold
                
            # Signal triggered! Set up order parameters
            # Map start index in ticks
            tick_idx = np.searchsorted(times, b_time)
            if tick_idx >= len(times):
                continue
                
            direction = brick['direction']
            if direction == 0:
                direction = -1
            
            bs = brick['brick_size']
            rr = rr_profiles[pred_c]
            
            # Setup Execution Geometry
            close_price = brick['close_price']
            bid_start = bids[tick_idx]
            ask_start = asks[tick_idx]
            spread = ask_start - bid_start
            
            if pred_c == 1: # T1 Market
                # Base execution on structural close price, but pay the actual live spread
                if direction == 1:
                    entry = close_price + spread
                    tp = entry + bs
                    sl = entry - bs
                else:
                    entry = close_price - spread
                    tp = entry - bs
                    sl = entry + bs
                fill_index = tick_idx
            elif pred_c == 2: # T2 Limit at Open
                # Seek open limit fill price
                limit_price = brick['open_price']
                filled = False
                fill_index = -1
                for j in range(tick_idx, len(times)):
                    tick_ask = asks[j]
                    tick_bid = bids[j]
                    if direction == 1 and tick_ask <= limit_price:
                        filled = True
                        fill_index = j
                        break
                    elif direction == -1 and tick_bid >= limit_price:
                        filled = True
                        fill_index = j
                        break
                    # If we hit SL before limit price, order never fills
                    # SL is open_price - direction * bs
                    limit_sl = limit_price - (direction * bs)
                    tick_price = bids[j] if direction == 1 else asks[j]
                    if (direction == 1 and tick_price <= limit_sl) or (direction == -1 and tick_price >= limit_sl):
                        break
                        
                if not filled:
                    continue # Never filled, no trade
                entry = limit_price
                tp = entry + (direction * 2.0 * bs)
                sl = entry - (direction * bs)
            elif pred_c == 3: # T3 Market Reversal
                # Enters opposite direction
                entry = asks[tick_idx] if direction == -1 else bids[tick_idx]
                tp = entry - (direction * 2.0 * bs)
                sl = entry + (direction * bs)
                direction = -direction # Invert execution direction
                fill_index = tick_idx
            elif pred_c == 4: # T4 Market Deep Reversal
                # Enters opposite direction
                entry = asks[tick_idx] if direction == -1 else bids[tick_idx]
                tp = entry - (direction * 3.0 * bs)
                sl = entry + (direction * bs)
                direction = -direction # Invert execution direction
                fill_index = tick_idx
                
            # Scan tick-by-tick for TP/SL touch starting from fill_index
            # Note: scan_exit_ticks starts at fill_index+1 to skip the fill tick itself
            is_win, exit_time = scan_exit_ticks(bids, asks, times, fill_index, direction, entry, tp, sl)
            pnl_R = rr if is_win else -1.0
            
            # Record the timestamp of this signal so we skip duplicate-timestamp bricks
            last_signal_brick_time = b_time
            
            active_trade = {
                "brick_id": brick['brick_id'],
                "setup_type": f"T{pred_c}",
                "direction": "BUY" if direction == 1 else "SELL",
                "entry_price": round(entry, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "entry_time": times[fill_index],
                "exit_time": exit_time,
                "pnl_R": pnl_R
            }
            
    # Force close final position if still open
    if active_trade is not None:
        trade_log.append(active_trade)
        
    # 4. Generate Reports
    logger.info("Generating reports...")
    trades_df = pd.DataFrame(trade_log)
    trades_df.to_csv(SIM_OUT_DIR / "sim_setup_classifier_trades.csv", index=False)
    logger.info(f"Saved trades log to {SIM_OUT_DIR}/sim_setup_classifier_trades.csv")
    
    daily_df = pd.DataFrame(list(daily_summaries.values()))
    daily_df.to_csv(SIM_OUT_DIR / "sim_setup_classifier_daily.csv", index=False)
    
    # Performance summary
    total_trades = len(trade_log)
    wins = sum(1 for t in trade_log if t['pnl_R'] > 0.0)
    losses = total_trades - wins
    win_rate = (wins / total_trades) if total_trades > 0 else 0.0
    total_return_R = sum(t['pnl_R'] for t in trade_log)
    
    # Save chart
    if total_trades > 0:
        cum_pnl = np.cumsum([t['pnl_R'] for t in trade_log])
        plt.figure(figsize=(10, 5))
        plt.plot(cum_pnl, label="SetupClassifier Cumulative Return (R)", color="green", linewidth=2.0)
        plt.axhline(0, color="red", linestyle="--", alpha=0.5)
        plt.xlabel("Trade Number")
        plt.ylabel("PnL (R)")
        plt.title(f"2026 Live Simulation - SetupClassifier Equity Curve (Return: +{total_return_R:.2f}R, WR: {win_rate:.2%})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        chart_path = SIM_OUT_DIR / "sim_setup_classifier_equity.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        logger.info(f"Saved equity curve to {chart_path}")
        
    # Markdown Report
    report = f"""# 2026 Live Simulation Report: SetupClassifier

This report details the simulated performance of the trained `SetupClassifier` replaying 2026 ticks out-of-sample under exact production conditions.

## Parameters
*   **K_MULTIPLIER:** {k_multiplier}
*   **Veto Threshold (P(T0)):** {veto_threshold}
*   **T1 Threshold:** {T1_threshold}
*   **T2 Threshold:** {T2_threshold} (Disabled)
*   **T3 Threshold:** {T3_threshold}
*   **T4 Threshold:** {T4_threshold}
*   **Daily Drawdown Stop:** $-5.0\\text{{ R}}$

## Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Total Trades Taken** | {total_trades} |
| **Wins** | {wins} |
| **Losses** | {losses} |
| **Win Rate** | **{win_rate:.2%}** |
| **Total Return (R)** | **+{total_return_R:.2f} R** |
| **Expectancy per Trade** | **{total_return_R/total_trades:+.4f} R** |
| **Total Observed Bricks** | {len(bricks_data)} |
| **EV per Observed Brick** | **{total_return_R/len(bricks_data):+.4f} R** |

{thresholds_table_md}
"""
    with open(SIM_OUT_DIR / "sim_setup_classifier_report.md", "w") as f:
        f.write(report)
        
    logger.info("=========================================")
    logger.info(" LIVE SIMULATION ENGINE COMPLETE")
    logger.info(f" Trades: {total_trades} | Wins: {wins} | Losses: {losses}")
    logger.info(f" Win Rate: {win_rate:.2%}")
    logger.info(f" Total Return: {total_return_R:+.2f} R")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
