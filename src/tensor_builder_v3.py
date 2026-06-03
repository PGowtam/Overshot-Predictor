"""
Phase 6: Tensor Builder V3
==========================
Constructs the final training arrays for the Exhaustion/Reversion objective.
Joins the new v3_labels.parquet (which contains the Exhaustion scalars)
with the existing micro and macro tensors from V1.
Splits data temporally (Train <= 2023, Val >= 2024) and saves as .npy
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
TENSOR_IN_DIR = BASE_DIR / "outputs" / "sim_labels" / "tensors"
V3_LBL_DIR = BASE_DIR / "outputs" / "sim_labels_v3"
OUT_DIR = BASE_DIR / "outputs" / "exec_tensors_v3"

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading V3 labels...")
    files = glob.glob(str(V3_LBL_DIR / "v3_labels_*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Drop rows with NaN in critical fields
    df = df.dropna(subset=['path_class', 'reversion_depth', 'ofi_peak', 'vel_peak', 'spread_current', 'wick_ratio', 'absorption_index'])
    
    # Deriving Labels
    df['y_reversion'] = df['path_class'].isin([1, 2]).astype(int)
    df['y_full_reversal'] = (df['path_class'] == 2).astype(int)
    df['strong_reversion'] = (df['reversion_depth'] > 1.5).astype(int)
    
    logger.info(f"Total valid samples: {len(df)}")
    
    # Temporal Split
    df['year'] = df['year'].astype(int)
    train_df = df[df['year'] <= 2023].copy()
    val_df = df[df['year'] >= 2024].copy()
    
    logger.info(f"Train split (<=2023): {len(train_df)}")
    logger.info(f"Val split (>=2024): {len(val_df)}")
    
    scalar_cols = ['spread_current', 'ofi_peak', 'ofi_slope', 'wick_ratio', 'absorption_index', 'vel_peak', 'vel_current']
    target_cols = ['y_reversion', 'y_full_reversal', 'strong_reversion', 'reversion_depth']
    
    def process_split(split_name, split_df):
        logger.info(f"Processing {split_name} split...")
        
        scalars_list = []
        macro_list = []
        micro_list = []
        targets_list = []
        
        valid_count = 0
        missing_count = 0
        
        # We iterate over the dataframe
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
            y, m, b = int(row['year']), int(row['month']), int(row['brick_id'])
            
            macro_path = TENSOR_IN_DIR / f"macro_{y}_{m:02d}_{b}.npy"
            micro_path = TENSOR_IN_DIR / f"micro_{y}_{m:02d}_{b}.npy"
            
            if macro_path.exists() and micro_path.exists():
                try:
                    macro = np.load(macro_path)
                    micro = np.load(micro_path)
                    
                    if macro.shape == (3,) and micro.shape == (100, 9):
                        macro_list.append(macro)
                        micro_list.append(micro)
                        
                        scalars_list.append(row[scalar_cols].values.astype(np.float32))
                        targets_list.append(row[target_cols].values.astype(np.float32))
                        valid_count += 1
                    else:
                        missing_count += 1
                except Exception:
                    missing_count += 1
            else:
                missing_count += 1
                
        logger.info(f"{split_name}: {valid_count} valid, {missing_count} missing/invalid tensors.")
        
        # Convert to numpy arrays
        X_scalars = np.stack(scalars_list)
        X_macro = np.stack(macro_list)
        X_micro = np.stack(micro_list)
        Y_targets = np.stack(targets_list)
        
        # Save to disk
        np.save(OUT_DIR / f"{split_name}_scalars.npy", X_scalars)
        np.save(OUT_DIR / f"{split_name}_macro.npy", X_macro)
        np.save(OUT_DIR / f"{split_name}_micro.npy", X_micro)
        np.save(OUT_DIR / f"{split_name}_targets.npy", Y_targets)
        
        logger.info(f"Saved {split_name} arrays to {OUT_DIR}")
        
    process_split("train", train_df)
    process_split("val", val_df)
    logger.info("Tensor building complete.")

if __name__ == "__main__":
    main()
