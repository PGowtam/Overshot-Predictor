import os
import glob
import logging
import argparse
import ctypes
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
import calendar

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MTPATSC_DIR = BASE_DIR / "MTPATSC"
DATA_DIR = BASE_DIR / "Data" / "Raw" / "Ticks"
OUTPUT_DIR = MTPATSC_DIR / "outputs" / "setup_classifier"

class LabelRow(ctypes.Structure):
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
        ("brick_duration_seconds", ctypes.c_int64)
    ]

lib_path = MTPATSC_DIR / "src" / "libmtpatsc_labels.dylib"
try:
    lib = ctypes.CDLL(str(lib_path))
    lib.generate_labels.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int)
    ]
    lib.generate_labels.restype = ctypes.POINTER(LabelRow)
    lib.free_labels.argtypes = [ctypes.POINTER(LabelRow)]
    lib.free_labels.restype = None
except Exception as e:
    logger.warning(f"Failed to load C++ library: {e}. Build it first.")

def process_chunk(year: int, month: int, files: list, target_start, target_end):
    log = logging.getLogger(f"Worker-{year}-{month:02d}")
    log.info(f"Processing chunk {year}-{month:02d}...")
    
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            pass
    if not dfs: return None
        
    df = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in df.columns and 'time_msc' not in df.columns:
        df['time_msc'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
        
    for col in ['bid', 'ask']:
        if col in df.columns:
            df[col] = df[col].astype(np.float64)
            
    df = df.sort_values('time_msc').reset_index(drop=True)
    
    bids = np.ascontiguousarray(df['bid'].values, dtype=np.float64)
    asks = np.ascontiguousarray(df['ask'].values, dtype=np.float64)
    times = np.ascontiguousarray(df['time_msc'].values, dtype=np.int64)
    num_ticks = len(bids)
    
    out_num_rows = ctypes.c_int(0)
    k_multiplier = 0.00118 # As per Phase 9 BrickOfTicks
    
    ptr = lib.generate_labels(bids, asks, times, num_ticks, k_multiplier, ctypes.byref(out_num_rows))
    
    if out_num_rows.value > 0 and ptr:
        data = []
        for i in range(out_num_rows.value):
            row = ptr[i]
            dt = datetime.utcfromtimestamp(row.timestamp / 1000.0).date()
            if target_start <= dt <= target_end:
                data.append({
                    "brick_id": row.brick_id,
                    "date": dt,
                    "timestamp": row.timestamp,
                    "direction": row.direction,
                    "C": row.close_price,
                    "O": row.open_price,
                    "K": row.brick_size,
                    "t1_win": row.t1_win,
                    "t1_y_mag": row.t1_y_mag,
                    "t2_win": row.t2_win,
                    "t2_y_mag": row.t2_y_mag,
                    "t2_filled": row.t2_filled,
                    "t3_win": row.t3_win,
                    "t3_y_mag": row.t3_y_mag,
                    "t4_win": row.t4_win,
                    "t4_y_mag": row.t4_y_mag,
                    "t4_filled": row.t4_filled,
                    "label": row.label,
                    "exclude_flag": row.exclude_flag,
                    "brick_duration_seconds": row.brick_duration_seconds,
                    "boundary_proximity": 0.0 
                })
        
        lib.free_labels(ptr)
        
        if data:
            df_out = pd.DataFrame(data)
            out_path = OUTPUT_DIR / f"labels_{year}_{month:02d}.parquet"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            df_out.to_parquet(out_path)
            log.info(f"✅ Saved {len(df_out)} MTPATSC labels to {out_path}")
            return len(df_out)
    return 0

def mp_worker(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()]
    )
    yr, mo, files, t_start, t_end = args
    return process_chunk(yr, mo, files, t_start, t_end)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    all_files = glob.glob(str(DATA_DIR / "**" / "*.parquet"), recursive=True)
    
    file_dict = {}
    for f in all_files:
        path = Path(f)
        day_str, month_str, year_str = path.stem, path.parent.name, path.parent.parent.name
        if day_str.isdigit() and month_str.isdigit() and year_str.isdigit():
            try:
                dt = datetime(int(year_str), int(month_str), int(day_str)).date()
                file_dict[dt] = f
            except ValueError:
                pass
                
    year_months = sorted(list(set((dt.year, dt.month) for dt in file_dict.keys())))
    logger.info(f"Found {len(year_months)} unique months of data to process.")
    
    tasks = []
    for y, m in year_months:
        target_start = datetime(y, m, 1).date()
        _, last_day = calendar.monthrange(y, m)
        target_end = datetime(y, m, last_day).date()
        
        # Load 7 days before target_start for the anchor optimization
        load_start = target_start - timedelta(days=7)
        load_end = target_end + timedelta(days=7) 
        
        chunk_files = []
        curr = load_start
        while curr <= load_end:
            if curr in file_dict:
                chunk_files.append(file_dict[curr])
            curr += timedelta(days=1)
            
        if chunk_files:
            tasks.append((y, m, chunk_files, target_start, target_end))
    
    if args.workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(mp_worker, tasks)
    else:
        results = [mp_worker(t) for t in tasks]
        
    total = sum(r for r in results if r is not None)
    logger.info(f"Finished generating all labels. Total valid setups: {total}")

if __name__ == "__main__":
    main()
