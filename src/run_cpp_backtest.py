import os
import sys
import ctypes
import argparse
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class CTrade(ctypes.Structure):
    _fields_ = [
        ("order_id", ctypes.c_int),
        ("direction", ctypes.c_int),
        ("original_limit", ctypes.c_double),
        ("limit_price", ctypes.c_double),
        ("tp_price", ctypes.c_double),
        ("sl_price", ctypes.c_double),
        ("brick_size", ctypes.c_double),
        ("delay_mins", ctypes.c_int),
        ("state", ctypes.c_int),
        ("created_t_msc", ctypes.c_int64),
        ("touch_t_msc", ctypes.c_int64),
        ("activation_t_msc", ctypes.c_int64),
        ("filled", ctypes.c_bool),
        ("fill_t_msc", ctypes.c_int64),
        ("exit_t_msc", ctypes.c_int64),
        ("pnl_R", ctypes.c_double),
        ("result_code", ctypes.c_int)
    ]

# Load C++ Library
lib_path = Path(__file__).parent.parent / "libengine.dylib"
if not lib_path.exists():
    raise FileNotFoundError("Compile libengine.dylib first!")

engine = ctypes.CDLL(str(lib_path))
engine.run_backtest_cpp.argtypes = [
    ctypes.POINTER(ctypes.c_double), # bids
    ctypes.POINTER(ctypes.c_double), # asks
    ctypes.POINTER(ctypes.c_int64),  # times_msc
    ctypes.c_int,                    # num_ticks
    ctypes.c_double,                 # k_multiplier
    ctypes.POINTER(ctypes.c_int),    # delays
    ctypes.c_int,                    # num_delays
    ctypes.POINTER(ctypes.c_int)     # out_num_trades
]
engine.run_backtest_cpp.restype = ctypes.POINTER(CTrade)
engine.free_trades.argtypes = [ctypes.POINTER(CTrade)]
engine.free_trades.restype = None

def process_year(year: int, data_dir: Path, delays: list, micro_k: float):
    year_dir = data_dir / str(year)
    if not year_dir.exists(): return []
    
    files = sorted(year_dir.rglob("*.parquet"))
    if not files: return []
    
    logger.info(f"[{year}] Loading {len(files)} parquet files...")
    df_list = []
    for f in files:
        df_list.append(pd.read_parquet(f))
    
    df = pd.concat(df_list, ignore_index=True)
    if 'time_msc' not in df.columns and 'timestamp' in df.columns:
        df['time_msc'] = df['timestamp'].astype('int64') // 10**6
    df = df.sort_values('time_msc')
    
    bids = np.ascontiguousarray(df['bid'].values, dtype=np.float64)
    asks = np.ascontiguousarray(df['ask'].values, dtype=np.float64)
    times = np.ascontiguousarray(df['time_msc'].values, dtype=np.int64)
    num_ticks = len(df)
    
    delays_arr = np.ascontiguousarray(delays, dtype=np.int32)
    num_delays = len(delays)
    
    out_num_trades = ctypes.c_int(0)
    
    logger.info(f"[{year}] Running C++ Engine on {num_ticks:,} ticks...")
    t0 = time.time()
    
    trades_ptr = engine.run_backtest_cpp(
        bids.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        asks.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        times.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        num_ticks,
        micro_k,
        delays_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        num_delays,
        ctypes.byref(out_num_trades)
    )
    
    n_trades = out_num_trades.value
    logger.info(f"[{year}] Engine finished in {time.time()-t0:.2f}s. Generated {n_trades} signals.")
    
    results = []
    if n_trades > 0:
        trades_array = ctypes.cast(trades_ptr, ctypes.POINTER(CTrade * n_trades)).contents
        for i in range(n_trades):
            t = trades_array[i]
            results.append({
                'year': year,
                'delay_mins': t.delay_mins,
                'filled': t.filled,
                'pnl_R': t.pnl_R,
                'result_code': t.result_code
            })
        
        engine.free_trades(trades_ptr)
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, default="2020,2021,2022,2023,2024,2025,2026")
    parser.add_argument("--delays", type=str, default="45")
    parser.add_argument("--micro-k", type=float, default=0.00236)
    args = parser.parse_args()
    
    years = [int(y.strip()) for y in args.years.split(",")]
    delays = [int(d.strip()) for d in args.delays.split(",")]
    
    data_dir = Path(__file__).parent.parent / "Data" / "Raw" / "Ticks"
    
    all_results = []
    for y in years:
        res = process_year(y, data_dir, delays, args.micro_k)
        all_results.extend(res)
        
    df_res = pd.DataFrame(all_results)
    if len(df_res) == 0:
        logger.info("No trades generated.")
        sys.exit(0)
        
    for delay in delays:
        delay_df = df_res[df_res['delay_mins'] == delay]
        filled = delay_df[delay_df['filled'] == True]
        wins = filled[filled['result_code'] == 0]
        losses = filled[filled['result_code'] == 1]
        invalidated = delay_df[delay_df['result_code'] == 3]
        
        tradeable = len(wins) + len(losses)
        if tradeable > 0:
            win_rate = len(wins) / tradeable * 100
            total_pnl = filled['pnl_R'].sum()
            avg_win = wins['pnl_R'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl_R'].mean() if len(losses) > 0 else 0
            
            logger.info(f"\n── 7-YEAR {delay}-MIN PULLBACK STABILIZATION RESULTS (2X Macro) ──")
            logger.info(f"Total Generated Signals: {len(delay_df)}")
            logger.info(f"Dodged Falling Knives:   {len(invalidated)} ({len(invalidated)/len(delay_df)*100:.1f}%)")
            logger.info(f"Trades Filled:           {tradeable}")
            logger.info(f"Win Rate:                {win_rate:.2f}%")
            logger.info(f"Expectancy:              {total_pnl/tradeable:+.4f} R")
            logger.info(f"Total P&L:               {total_pnl:+.1f} R")
            logger.info(f"Avg Win:                 {avg_win:+.2f} R")
            logger.info(f"Avg Loss:                {avg_loss:+.2f} R")
