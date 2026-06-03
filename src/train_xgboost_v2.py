"""
Train XGBoost V2 (Validation Step)
==================================
Validates the predictive lift of the new Macro and Summary features 
before implementing the full Transformer network.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
TENSOR_DIR = BASE_DIR / "outputs" / "exec_tensors_v2"
OUTPUT_DIR = BASE_DIR / "outputs" / "xgboost_v2"

logger = logging.getLogger(__name__)

def load_data(split_name):
    logger.info(f"Loading {split_name} data...")
    macro = np.load(TENSOR_DIR / f"{split_name}_macro.npy")     # (N, 10, 11)
    summary = np.load(TENSOR_DIR / f"{split_name}_summary.npy") # (N, 5)
    y_class = np.load(TENSOR_DIR / f"{split_name}_y_class.npy") # (N, 1)
    y_mag = np.load(TENSOR_DIR / f"{split_name}_y_mag.npy")     # (N, 1)
    
    # Flatten macro temporal dimension for XGBoost: 10 * 11 = 110 features
    N = macro.shape[0]
    macro_flat = macro.reshape((N, -1))
    
    # Concatenate macro and summary
    X = np.concatenate([macro_flat, summary], axis=1) # (N, 115)
    y = y_class.flatten()
    y_reg = y_mag.flatten()
    
    # Filter out NaNs if any
    valid_mask = ~np.isnan(y)
    return X[valid_mask], y[valid_mask], y_reg[valid_mask]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / "xgb_train.log"), mode='w')
        ]
    )
    
    # 1. Load Data
    X_train, y_train, _ = load_data("train")
    X_val, y_val, y_val_reg = load_data("val")
    
    logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Val shapes: X={X_val.shape}, y={y_val.shape}")
    
    # 2. Train XGBoost
    logger.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        tree_method='hist',
        scale_pos_weight=1.85, # 65/35 imbalance correction
        random_state=42,
        early_stopping_rounds=20
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    
    # 3. Evaluate
    logger.info("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, val_probs)
    logger.info(f"ROC AUC: {auc:.4f}")
    
    # Top 10% Precision Check
    threshold_90 = np.percentile(val_probs, 90)
    top_10_mask = val_probs >= threshold_90
    
    y_val_top_10 = y_val[top_10_mask]
    precision_top_10 = np.mean(y_val_top_10)
    
    logger.info(f"Precision @ Top 10% predictions (Threshold >= {threshold_90:.4f}): {precision_top_10*100:.2f}%")
    
    # Profit Factor Proxy
    # Assuming 1:1 RR, a win is +1, loss is -1.
    wins = np.sum(y_val_top_10 == 1)
    losses = np.sum(y_val_top_10 == 0)
    profit_factor = wins / (losses + 1e-8)
    
    logger.info(f"Profit Factor @ Top 10%: {profit_factor:.2f} ({wins}W / {losses}L)")
    
    # Save Model
    model.save_model(OUTPUT_DIR / "xgboost_v2.json")
    logger.info("Saved XGBoost model to outputs/xgboost_v2/")

if __name__ == "__main__":
    main()
