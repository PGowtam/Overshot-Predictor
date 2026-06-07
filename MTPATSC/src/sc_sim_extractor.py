import os
import sys
import ctypes
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Extractor")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MTPATSC_DIR = BASE_DIR / "MTPATSC"
OUTPUT_DIR = MTPATSC_DIR / "outputs" / "setup_classifier"

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

def main():
    if len(sys.argv) > 2:
        ticks_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])
    else:
        ticks_path = BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"
        if not ticks_path.exists():
            ticks_path = BASE_DIR / "data" / "xauusd_ticks_2026.parquet"
        out_path = OUTPUT_DIR / "sim_2026_bricks.parquet"
        
    if not ticks_path.exists():
        logger.error(f"Ticks parquet not found at {ticks_path}")
        sys.exit(1)
            
    logger.info("Loading 2026 ticks...")
    df_ticks = pd.read_parquet(ticks_path)
    if 'timestamp' in df_ticks.columns and 'time_msc' not in df_ticks.columns:
        df_ticks['time_msc'] = pd.to_datetime(df_ticks['timestamp']).astype('int64') // 10**6
    df_ticks = df_ticks.sort_values('time_msc').reset_index(drop=True)
    
    bids = np.ascontiguousarray(df_ticks['bid'].values, dtype=np.float64)
    asks = np.ascontiguousarray(df_ticks['ask'].values, dtype=np.float64)
    times = np.ascontiguousarray(df_ticks['time_msc'].values, dtype=np.int64)
    
    lib_path = MTPATSC_DIR / "src" / "libmtpatsc_engine.dylib"
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
    
    logger.info("Executing C++ Feature Engine...")
    out_num_rows = ctypes.c_int(0)
    k_multiplier = 0.00118
    ptr = lib.generate_dataset(bids, asks, times, len(bids), k_multiplier, ctypes.byref(out_num_rows))
    
    data = []
    if out_num_rows.value > 0 and ptr:
        for i in range(out_num_rows.value):
            row = ptr[i]
            if row.exclude_flag != 1:
                # Store serialized lists to prevent complex parquet casting
                data.append({
                    "brick_id": row.brick_id,
                    "timestamp": row.timestamp,
                    "direction": row.direction,
                    "close_price": row.close_price,
                    "open_price": row.open_price,
                    "brick_size": row.brick_size,
                    "t1_win": int(row.t1_win),
                    "t2_win": int(row.t2_win),
                    "t3_win": int(row.t3_win),
                    "t4_win": int(row.t4_win),
                    "label": int(row.label),
                    "ancs_fine": np.nan_to_num(np.array(row.ancs_fine, dtype=np.float32).reshape(10, 6)).tolist(),
                    "ancs_coarse": np.nan_to_num(np.array(row.ancs_coarse, dtype=np.float32).reshape(5, 6)).tolist(),
                    "history": np.nan_to_num(np.array(row.history, dtype=np.float32).reshape(5, 5, 6)).tolist(),
                    "candle_features": np.nan_to_num(np.array(row.candle_features, dtype=np.float32)).tolist(),
                    "momentum": np.nan_to_num(np.array(row.momentum, dtype=np.float32)).tolist()
                })
        lib.free_dataset(ptr)
        
    df_out = pd.DataFrame(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, engine='pyarrow')
    logger.info(f"Saved {len(df_out)} valid bricks to {out_path}")

if __name__ == "__main__":
    main()
