import os
import sys
import ctypes
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

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
    ctypes.POINTER(ctypes.c_int),    
    ctypes.c_int,                    
    ctypes.POINTER(ctypes.c_int)     
]
engine.run_backtest_reversal.restype = ctypes.POINTER(CTrade)
engine.free_trades.argtypes = [ctypes.POINTER(CTrade)]
engine.free_trades.restype = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, default="2020,2021,2022,2023,2024,2025,2026")
    parser.add_argument("--delays", type=str, default="5,10,15,20,25,30")
    parser.add_argument("--micro-k", type=float, default=0.00236)
    args = parser.parse_args()

    years = [y.strip() for y in args.years.split(',')]
    delays_list = [int(d.strip()) for d in args.delays.split(',')]
    c_delays = (ctypes.c_int * len(delays_list))(*delays_list)
    micro_k = args.micro_k

    data_base = Path(__file__).parent.parent / "Data" / "Raw" / "Ticks"

    all_results = []

    for year in years:
        year_dir = data_base / year
        files = sorted(year_dir.rglob("*.parquet"))
        if not files: continue

        print(f"[{year}] Loading {len(files)} parquet files...")
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
        
        print(f"[{year}] Running Reversal C++ Engine on {num_ticks:,} ticks...")
        out_num_trades = ctypes.c_int(0)
        t0 = time.time()
        
        trades_ptr = engine.run_backtest_reversal(
            bids.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            asks.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            times.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            num_ticks,
            micro_k,
            c_delays,
            len(delays_list),
            ctypes.byref(out_num_trades)
        )
        
        n_trades = out_num_trades.value
        print(f"[{year}] Engine finished in {time.time()-t0:.2f}s. Generated {n_trades} signals.")

        if n_trades > 0:
            trades_array = ctypes.cast(trades_ptr, ctypes.POINTER(CTrade * n_trades)).contents
            for i in range(n_trades):
                t = trades_array[i]
                if t.result_code != -1:
                    all_results.append({
                        'delay': t.delay_mins,
                        'result_code': t.result_code,
                        'pnl_R': t.pnl_R
                    })
            engine.free_trades(trades_ptr)

    if not all_results:
        print("No trades generated.")
        return

    res_df = pd.DataFrame(all_results)
    
    print(f"\n── 7-YEAR FULL REVERSAL SWEEP RESULTS ({micro_k} Bricks) ──")
    for delay in delays_list:
        ddf = res_df[res_df['delay'] == delay]
        
        dodged = len(ddf[ddf['result_code'] == 3])
        missed = len(ddf[ddf['result_code'] == 2])
        filled_df = ddf[ddf['result_code'].isin([0, 1])]
        
        trades_filled = len(filled_df)
        if trades_filled == 0:
            print(f"Delay {delay}m: No trades filled.")
            continue
            
        wins = len(filled_df[filled_df['result_code'] == 0])
        win_rate = wins / trades_filled * 100
        expectancy = filled_df['pnl_R'].mean()
        total_pnl = filled_df['pnl_R'].sum()
        
        print(f"Delay {delay:02d}m -> Filled: {trades_filled:<5} | Dodged Losses: {dodged:<5} | Missed Wins: {missed:<5} | WR: {win_rate:.2f}% | Exp: {expectancy:+.4f} R | PnL: {total_pnl:+.1f} R")

if __name__ == "__main__":
    main()
