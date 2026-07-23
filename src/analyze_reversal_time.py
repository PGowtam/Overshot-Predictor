import os
import sys
import ctypes
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

lib_path = Path(__file__).parent.parent / "libengine_reversal.dylib"
engine = ctypes.CDLL(str(lib_path))
engine.run_backtest_reversal.argtypes = [
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_int64),  
    ctypes.c_int,                    
    ctypes.c_double,                 
    ctypes.POINTER(ctypes.c_int)     
]
engine.run_backtest_reversal.restype = ctypes.POINTER(CTrade)
engine.free_trades.argtypes = [ctypes.POINTER(CTrade)]
engine.free_trades.restype = None

def main():
    data_dir = Path(__file__).parent.parent / "Data" / "Raw" / "Ticks" / "2026"
    files = sorted(data_dir.rglob("*.parquet"))
    
    df_list = []
    for f in files: df_list.append(pd.read_parquet(f))
    df = pd.concat(df_list, ignore_index=True)
    if 'time_msc' not in df.columns and 'timestamp' in df.columns:
        df['time_msc'] = df['timestamp'].astype('int64') // 10**6
    df = df.sort_values('time_msc')
    
    bids = np.ascontiguousarray(df['bid'].values, dtype=np.float64)
    asks = np.ascontiguousarray(df['ask'].values, dtype=np.float64)
    times = np.ascontiguousarray(df['time_msc'].values, dtype=np.int64)
    num_ticks = len(df)
    
    micro_k = 0.00236
    out_num_trades = ctypes.c_int(0)
    
    logger.info("Running C++ Reversal Engine...")
    t0 = time.time()
    trades_ptr = engine.run_backtest_reversal(
        bids.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        asks.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        times.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        num_ticks,
        micro_k,
        ctypes.byref(out_num_trades)
    )
    n_trades = out_num_trades.value
    logger.info(f"Engine finished in {time.time()-t0:.2f}s. Generated {n_trades} signals.")
    
    if n_trades > 0:
        trades_array = ctypes.cast(trades_ptr, ctypes.POINTER(CTrade * n_trades)).contents
        results = []
        for i in range(n_trades):
            t = trades_array[i]
            if t.result_code in [0, 1] and t.filled:
                duration_mins = (t.exit_t_msc - t.fill_t_msc) / 60000.0
                results.append({
                    'result': 'win' if t.result_code == 0 else 'loss',
                    'duration_mins': duration_mins,
                    'pnl_R': t.pnl_R
                })
        
        engine.free_trades(trades_ptr)
        
        df_res = pd.DataFrame(results)
        wins = df_res[df_res['result'] == 'win']['duration_mins']
        losses = df_res[df_res['result'] == 'loss']['duration_mins']
        
        logger.info("\n── BASELINE REVERSAL (0-DELAY) DURATION STATS (2026) ──")
        logger.info(f"Total Trades: {len(df_res)}")
        logger.info(f"Win Rate:     {len(wins)/len(df_res)*100:.2f}%")
        logger.info(f"Expectancy:   {df_res['pnl_R'].mean():+.4f} R")
        
        logger.info("\n[WIN DURATION]")
        logger.info(f"Median: {wins.median():.2f} mins")
        logger.info(f"75th %: {wins.quantile(0.75):.2f} mins")
        logger.info(f"90th %: {wins.quantile(0.90):.2f} mins")
        
        logger.info("\n[LOSS DURATION]")
        logger.info(f"Median: {losses.median():.2f} mins")
        logger.info(f"75th %: {losses.quantile(0.75):.2f} mins")
        logger.info(f"90th %: {losses.quantile(0.90):.2f} mins")

if __name__ == "__main__":
    main()
