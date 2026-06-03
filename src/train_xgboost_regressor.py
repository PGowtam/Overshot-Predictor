"""
Train XGBoost Regressor (Experiment)
====================================
Tests whether the raw sequences (100x9 micro + 10x11 macro) contain signal 
by predicting the smoother y_reg (Overshoot) target instead of y_class.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
TENSOR_DIR = BASE_DIR / "outputs" / "exec_tensors_v2"
OUTPUT_DIR = BASE_DIR / "outputs" / "xgboost_regressor"

logger = logging.getLogger(__name__)

def load_data_flat(split_name):
    logger.info(f"Loading {split_name} data...")
    micro = np.load(TENSOR_DIR / f"{split_name}_micro.npy")     # (N, 100, 9)
    macro = np.load(TENSOR_DIR / f"{split_name}_macro.npy")     # (N, 10, 11)
    y_mag = np.load(TENSOR_DIR / f"{split_name}_y_mag.npy")     # (N, 1)
    
    N = micro.shape[0]
    
    # Flatten micro: 100 * 9 = 900 features
    micro_flat = micro.reshape((N, -1))
    
    # Flatten macro: 10 * 11 = 110 features
    macro_flat = macro.reshape((N, -1))
    
    # Total: 1010 features
    X = np.concatenate([micro_flat, macro_flat], axis=1) # (N, 1010)
    y_reg = y_mag.flatten()
    
    # Filter out NaNs if any
    valid_mask = ~np.isnan(y_reg)
    return X[valid_mask], y_reg[valid_mask]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / "xgb_regressor_train.log"), mode='w')
        ]
    )
    
    # 1. Load Data
    X_train, y_train_reg = load_data_flat("train")
    X_val, y_val_reg = load_data_flat("val")
    
    logger.info(f"Train shapes: X={X_train.shape}, y_reg={y_train_reg.shape}")
    logger.info(f"Val shapes: X={X_val.shape}, y_reg={y_val_reg.shape}")
    
    # 2. Train XGBoost Regressor
    logger.info("Training XGBoost Regressor (Flattened Sequence)...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6, 
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        early_stopping_rounds=20
    )
    
    model.fit(
        X_train, y_train_reg,
        eval_set=[(X_val, y_val_reg)],
        verbose=10
    )
    
    # 3. Evaluate
    logger.info("Evaluating on Validation Set...")
    val_preds = model.predict(X_val)
    
    pearson_corr, _ = pearsonr(y_val_reg, val_preds)
    spearman_corr, _ = spearmanr(y_val_reg, val_preds)
    
    logger.info(f"Pearson Correlation: {pearson_corr:.4f}")
    logger.info(f"Spearman Rank Correlation: {spearman_corr:.4f}")
    
    # Top-Decile Overshoot Check
    threshold_90 = np.percentile(val_preds, 90)
    top_10_mask = val_preds >= threshold_90
    
    actual_overshoot_top_10 = y_val_reg[top_10_mask]
    avg_overshoot_top_10 = np.mean(actual_overshoot_top_10)
    baseline_overshoot = np.mean(y_val_reg)
    
    logger.info(f"Baseline Avg Overshoot: {baseline_overshoot:.4f}")
    logger.info(f"Avg Actual Overshoot @ Top 10% predictions (Pred >= {threshold_90:.4f}): {avg_overshoot_top_10:.4f}")
    
    # Save Model
    model.save_model(OUTPUT_DIR / "xgboost_regressor.json")
    logger.info("Saved XGBoost Regressor model to outputs/xgboost_regressor/")

if __name__ == "__main__":
    main()
