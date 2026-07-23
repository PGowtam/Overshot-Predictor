"""
Tensor Builder V2 (Dynamic Extraction)
======================================
Builds the 100x9 Micro, 10x11 Macro, and 5D Summary tensors directly from 
existing label parquets and micro numpy arrays.
"""

import os
import glob
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_DIR = BASE_DIR / "outputs" / "sim_labels"
TENSOR_DIR = LABELS_DIR / "tensors"

OUTPUT_DIR = BASE_DIR / "outputs" / "exec_tensors_v2"

logger = logging.getLogger(__name__)

# ── Split Boundaries ─────────────────────────────────────────────
TRAIN_END = pd.Timestamp("2025-06-30 23:59:59", tz="UTC")
VAL_END   = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")


def sequence_to_array(seq_str: str) -> np.ndarray:
    arr = np.zeros(100, dtype=np.float32)
    seq_str = seq_str[-100:]
    start_idx = 100 - len(seq_str)
    for i, char in enumerate(seq_str):
        if char == '1':
            arr[start_idx + i] = 1.0
        elif char == '0':
            arr[start_idx + i] = -1.0
    return arr


def build_split(split_name: str, df: pd.DataFrame, undersample: bool = False):
    logger.info(f"Building {split_name} split with {len(df):,} samples...")
    
    n_samples = len(df)
    
    X_micro = np.zeros((n_samples, 100, 9), dtype=np.float32)
    X_macro = np.zeros((n_samples, 10, 11), dtype=np.float32)
    X_summary = np.zeros((n_samples, 5), dtype=np.float32)
    
    y_class = np.zeros((n_samples, 1), dtype=np.float32)
    y_mag = np.zeros((n_samples, 1), dtype=np.float32)
    
    valid_count = 0
    missing = 0
    
    macro_q = deque(maxlen=10)
    
    x_range = np.arange(100)
    x_mean = 49.5
    var_x = 833.25
    x_diff = x_range - x_mean
    
    t_start = time.time()
    for i, row in enumerate(df.itertuples()):
        b_id = row.brick_id
        year = getattr(row, 'year', None)
        month = getattr(row, 'month', None)
        
        if year is not None and month is not None:
            month_str = f"{int(month):02d}"
            micro_path = TENSOR_DIR / f"micro_{year}_{month_str}_{b_id}.npy"
            macro_path = TENSOR_DIR / f"macro_{year}_{month_str}_{b_id}.npy"
        elif year is not None:
            micro_path = TENSOR_DIR / f"micro_{year}_{b_id}.npy"
            macro_path = TENSOR_DIR / f"macro_{year}_{b_id}.npy"
        else:
            micro_path = TENSOR_DIR / f"micro_{b_id}.npy"
            macro_path = TENSOR_DIR / f"macro_{b_id}.npy"
        
        if not (micro_path.exists() and macro_path.exists()):
            missing += 1
            continue
            
        try:
            # Micro remains 100x9
            micro_data = np.load(micro_path)
            
            # --- 5D Summary Extraction ---
            # 1. directional_entropy
            progress = micro_data[:, 5]
            diff_prog = np.diff(progress)
            up_ticks = np.sum(diff_prog > 0)
            total_moves = np.sum(diff_prog != 0)
            p_up = up_ticks / total_moves if total_moves > 0 else 0.5
            if p_up == 0 or p_up == 1:
                directional_entropy = 0.0
            else:
                directional_entropy = -p_up * np.log(p_up) - (1 - p_up) * np.log(1 - p_up)
                
            # 2. tick_burst (Proxy: average z_vel of last 5 ticks)
            tick_burst = np.mean(micro_data[-5:, 3])
            
            # 3. ofi_accel
            z_ofi = micro_data[:, 0]
            ofi_accel = z_ofi[-1] - z_ofi[-11]
            
            # 4. ofi_curvature
            ofi_curvature = ofi_accel - (z_ofi[-11] - z_ofi[-21])
            
            # 5. depth_slope
            z_depth = micro_data[:, 1]
            cov = np.sum(x_diff * (z_depth - np.mean(z_depth)))
            depth_slope = cov / 83325.0
            
            summary_vec = np.array([
                directional_entropy, tick_burst, ofi_accel, ofi_curvature, depth_slope
            ], dtype=np.float32)
            
            # --- 11D Macro Extraction ---
            orig_macro = np.load(macro_path) # [log_dur, direction, z_size]
            
            enriched_macro = np.zeros(11, dtype=np.float32)
            enriched_macro[0:3] = orig_macro
            enriched_macro[3] = row.brick_entropy_10
            enriched_macro[4] = row.persistence_10
            enriched_macro[5] = row.duration_compression
            enriched_macro[6] = row.vol_regime
            enriched_macro[7] = row.hour_sin
            enriched_macro[8] = row.hour_cos
            enriched_macro[9] = row.brick_in_run_log
            enriched_macro[10] = row.local_wr
            
            macro_q.append(enriched_macro)
            
            if len(macro_q) == 10:
                X_micro[valid_count] = micro_data
                X_macro[valid_count] = np.array(macro_q)
                X_summary[valid_count] = summary_vec
                
                y_class[valid_count, 0] = row.y_class
                y_mag[valid_count, 0] = row.y_mag
                
                valid_count += 1
            
        except Exception as e:
            logger.error(f"Error loading brick {b_id}: {e}")
            missing += 1
            
        if (i + 1) % 10000 == 0:
            logger.info(f"  Processed {i+1}/{n_samples}...")
            
    logger.info(f"Completed {split_name}: {valid_count} valid, {missing} missing. Time: {time.time()-t_start:.1f}s")
    
    # Truncate
    X_micro = X_micro[:valid_count]
    X_macro = X_macro[:valid_count]
    X_summary = X_summary[:valid_count]
    y_class = y_class[:valid_count]
    y_mag = y_mag[:valid_count]
    
    np.save(OUTPUT_DIR / f"{split_name}_micro.npy", X_micro)
    np.save(OUTPUT_DIR / f"{split_name}_macro.npy", X_macro)
    np.save(OUTPUT_DIR / f"{split_name}_summary.npy", X_summary)
    np.save(OUTPUT_DIR / f"{split_name}_y_class.npy", y_class)
    np.save(OUTPUT_DIR / f"{split_name}_y_mag.npy", y_mag)
    
    logger.info(f"Saved {split_name} arrays to {OUTPUT_DIR}/")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    logger.info("Loading label parquets...")
    label_files = glob.glob(str(LABELS_DIR / "sim_labels_*.parquet"))
    
    if not label_files:
        logger.error(f"No label files found in {LABELS_DIR}")
        return
        
    dfs = [pd.read_parquet(f, engine='fastparquet') for f in label_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df):,} labels.")
    
    if 'utc_time' not in df.columns:
        df['utc_time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('utc_time').reset_index(drop=True)
    
    # ── Compute Dynamic Features ──
    logger.info("Computing Pandas Dynamic Features...")
    
    # 1. duration
    df['duration'] = df['utc_time'].diff().dt.total_seconds().fillna(0)
    
    # 2. ema_duration_20 & duration_compression
    ema_duration_20 = df['duration'].ewm(span=20, adjust=False).mean()
    df['duration_compression'] = (ema_duration_20 / (df['duration'] + 1e-8)).astype(np.float32)
    
    # 3. vol_regime (std / mean)
    roll_dur = df['duration'].rolling(window=20, min_periods=1)
    df['vol_regime'] = (roll_dur.std() / (roll_dur.mean() + 1e-8)).fillna(0).astype(np.float32)
    
    # 4. brick_entropy_10
    def calc_entropy(x):
        p_up = np.sum(x == 1) / len(x)
        if p_up == 0 or p_up == 1: return 0.0
        return -p_up * np.log(p_up) - (1 - p_up) * np.log(1 - p_up)
    
    df['brick_entropy_10'] = df['direction'].rolling(window=10, min_periods=1).apply(calc_entropy, raw=True).fillna(0).astype(np.float32)
    
    # 5. persistence_10
    def calc_persistence(x):
        max_c = 1
        curr_c = 1
        for i in range(1, len(x)):
            if x[i] == x[i-1]:
                curr_c += 1
                max_c = max(max_c, curr_c)
            else:
                curr_c = 1
        return max_c / 10.0
        
    df['persistence_10'] = df['direction'].rolling(window=10, min_periods=1).apply(calc_persistence, raw=True).fillna(0).astype(np.float32)
    
    # 6. session sin/cos
    hour = df['utc_time'].dt.hour.values
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    
    # 7. brick_in_run
    direction = df['direction'].values
    N = len(df)
    brick_in_run = np.ones(N, dtype=np.float32)
    for i in range(1, N):
        if direction[i] == direction[i-1]:
            brick_in_run[i] = brick_in_run[i-1] + 1
    df['brick_in_run_log'] = np.log1p(brick_in_run).astype(np.float32)
    
    # 8. local_wr
    y = df['y_class'].values.astype(np.float32)
    local_wr = np.full(N, 0.3533, dtype=np.float32)
    for i in range(20, N):
        window = y[i-20:i]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            local_wr[i] = np.mean(valid_window)
    df['local_wr'] = local_wr

    logger.info("Features computed.")
    
    # ── Split ──
    train_mask = df['utc_time'] <= TRAIN_END
    val_mask = (df['utc_time'] > TRAIN_END) & (df['utc_time'] <= VAL_END)
    test_mask = df['utc_time'] > VAL_END
    
    df_train = df[train_mask]
    df_val = df[val_mask]
    df_test = df[test_mask]
    
    logger.info(f"Split sizes -> Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")
    
    build_split("train", df_train)
    build_split("val", df_val)
    build_split("test", df_test)

if __name__ == "__main__":
    main()
