import os
import ctypes
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

class CFeatureRow(ctypes.Structure):
    _fields_ = [
        ("brick_id", ctypes.c_int),
        ("timestamp", ctypes.c_int64),
        ("direction", ctypes.c_int),
        ("entry_price", ctypes.c_double),
        ("brick_size", ctypes.c_double),
        
        ("time_sin", ctypes.c_double),
        ("time_cos", ctypes.c_double),
        
        ("ema_50_5m_dist", ctypes.c_double),
        ("ema_200_5m_dist", ctypes.c_double),
        ("atr_14_5m", ctypes.c_double),
        ("return_12_5m", ctypes.c_double),
        
        ("ema_50_15m_dist", ctypes.c_double),
        ("ema_200_15m_dist", ctypes.c_double),
        ("atr_14_15m", ctypes.c_double),
        ("return_4_15m", ctypes.c_double),
        
        ("label_t1", ctypes.c_int),
        ("label_t2", ctypes.c_int),
        ("label_t3", ctypes.c_int),
        ("label_t4", ctypes.c_int)
    ]

lib_path = Path(__file__).parent.parent / "libfeature_engine_v4.dylib"
engine = ctypes.CDLL(str(lib_path))
engine.generate_hybrid_features.argtypes = [
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_int64),  
    ctypes.c_int,                    
    ctypes.c_double,                 
    ctypes.POINTER(ctypes.c_int)     
]
engine.generate_hybrid_features.restype = ctypes.POINTER(CFeatureRow)
engine.free_features.argtypes = [ctypes.POINTER(CFeatureRow)]
engine.free_features.restype = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, default="2026")
    parser.add_argument("--micro-k", type=float, default=0.00118)
    args = parser.parse_args()

    years = [y.strip() for y in args.years.split(',')]
    micro_k = args.micro_k

    data_base = Path(__file__).parent.parent / "Data" / "Raw" / "Ticks"
    out_dir = Path(__file__).parent.parent / "outputs" / "sim_labels_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        
        print(f"[{year}] Computing Price Action & N-Grams on {num_ticks:,} ticks via C++ Engine...")
        out_num_rows = ctypes.c_int(0)
        t0 = time.time()
        
        features_ptr = engine.generate_hybrid_features(
            bids.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            asks.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            times.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            num_ticks,
            micro_k,
            ctypes.byref(out_num_rows)
        )
        
        n_rows = out_num_rows.value
        print(f"[{year}] C++ Engine finished in {time.time()-t0:.2f}s. Generated {n_rows} clean feature rows.")

        if n_rows > 0:
            features_array = ctypes.cast(features_ptr, ctypes.POINTER(CFeatureRow * n_rows)).contents
            
            # Efficiently convert struct array to dict of lists
            data = {
                "brick_id": np.zeros(n_rows, dtype=np.int32),
                "timestamp": np.zeros(n_rows, dtype=np.int64),
                "direction": np.zeros(n_rows, dtype=np.int32),
                "entry_price": np.zeros(n_rows, dtype=np.float64),
                "brick_size": np.zeros(n_rows, dtype=np.float64),
                
                "time_sin": np.zeros(n_rows, dtype=np.float64),
                "time_cos": np.zeros(n_rows, dtype=np.float64),
                
                "ema_50_5m_dist": np.zeros(n_rows, dtype=np.float64),
                "ema_200_5m_dist": np.zeros(n_rows, dtype=np.float64),
                "atr_14_5m": np.zeros(n_rows, dtype=np.float64),
                "return_12_5m": np.zeros(n_rows, dtype=np.float64),
                
                "ema_50_15m_dist": np.zeros(n_rows, dtype=np.float64),
                "ema_200_15m_dist": np.zeros(n_rows, dtype=np.float64),
                "atr_14_15m": np.zeros(n_rows, dtype=np.float64),
                "return_4_15m": np.zeros(n_rows, dtype=np.float64),
                
                "label_t1": np.zeros(n_rows, dtype=np.int32),
                "label_t2": np.zeros(n_rows, dtype=np.int32),
                "label_t3": np.zeros(n_rows, dtype=np.int32),
                "label_t4": np.zeros(n_rows, dtype=np.int32)
            }
            
            for i in range(n_rows):
                r = features_array[i]
                data["brick_id"][i] = r.brick_id
                data["timestamp"][i] = r.timestamp
                data["direction"][i] = r.direction
                data["entry_price"][i] = r.entry_price
                data["brick_size"][i] = r.brick_size
                data["time_sin"][i] = r.time_sin
                data["time_cos"][i] = r.time_cos
                data["ema_50_5m_dist"][i] = r.ema_50_5m_dist
                data["ema_200_5m_dist"][i] = r.ema_200_5m_dist
                data["atr_14_5m"][i] = r.atr_14_5m
                data["return_12_5m"][i] = r.return_12_5m
                data["ema_50_15m_dist"][i] = r.ema_50_15m_dist
                data["ema_200_15m_dist"][i] = r.ema_200_15m_dist
                data["atr_14_15m"][i] = r.atr_14_15m
                data["return_4_15m"][i] = r.return_4_15m
                data["label_t1"][i] = r.label_t1
                data["label_t2"][i] = r.label_t2
                data["label_t3"][i] = r.label_t3
                data["label_t4"][i] = r.label_t4

            engine.free_features(features_ptr)
            
            res_df = pd.DataFrame(data)
            
            out_file = out_dir / f"v4_features_{year}.parquet"
            res_df.to_parquet(out_file)
            print(f"[{year}] Saved to {out_file}\n")
            
            valid_t1 = res_df[res_df['label_t1'] != -1]
            valid_t2 = res_df[res_df['label_t2'] >= 0] # Exclude -2 (not triggered) and -1 (timeout)
            valid_t3 = res_df[res_df['label_t3'] != -1]
            valid_t4 = res_df[res_df['label_t4'] != -1]

            wr_t1 = (valid_t1['label_t1'] == 1).mean() * 100 if len(valid_t1) > 0 else 0
            wr_t2 = (valid_t2['label_t2'] == 1).mean() * 100 if len(valid_t2) > 0 else 0
            wr_t3 = (valid_t3['label_t3'] == 1).mean() * 100 if len(valid_t3) > 0 else 0
            wr_t4 = (valid_t4['label_t4'] == 1).mean() * 100 if len(valid_t4) > 0 else 0
            
            print(f"── QUICK STATS ({year}) ──")
            print(f"Total Bricks: {len(res_df)}")
            print(f"T1 (1:1 Cont) Win Rate:     {wr_t1:.2f}%")
            print(f"T2 (1:2 Pullback) Win Rate: {wr_t2:.2f}% (Triggered {len(valid_t2)/len(res_df)*100:.1f}%)")
            print(f"T3 (1:2 Reversal) Win Rate: {wr_t3:.2f}%")
            print(f"T4 (1:3 Deep Rev) Win Rate: {wr_t4:.2f}%")

if __name__ == "__main__":
    main()
