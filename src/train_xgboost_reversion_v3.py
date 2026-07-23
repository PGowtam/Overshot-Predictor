"""
Phase 4 & 5: XGBoost Baseline Testing for Reversion
===================================================
Trains 3 progressively harder models (Model A: Scalars, Model B: +Macro, Model C: +Micro)
on the 3 reversion targets (Reversion, Full Reversal, Strong Reversion).
Evaluates Top 10% Reversion Rate and Profit Factor.
"""

import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from pathlib import Path
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
TENSOR_DIR = BASE_DIR / "outputs" / "exec_tensors_v3"

logger = logging.getLogger(__name__)

def evaluate_model(model, X_val, y_val, target_name, model_name):
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    
    # Top 10% Decile metrics
    top_10_idx = np.argsort(preds)[-len(preds)//10:]
    top_10_y = y_val[top_10_idx]
    
    wins = np.sum(top_10_y)
    losses = len(top_10_y) - wins
    hit_rate = (wins / len(top_10_y)) * 100
    pf = wins / (losses + 1e-8)
    
    logger.info(f"[{model_name}] {target_name} -> AUC: {auc:.4f} | Top 10% Hit Rate: {hit_rate:.2f}% | PF: {pf:.2f}")
    return auc, hit_rate, pf

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Loading V3 Tensors...")
    
    try:
        X_train_scalars = np.load(TENSOR_DIR / "train_scalars.npy")
        X_train_macro = np.load(TENSOR_DIR / "train_macro.npy").reshape(len(X_train_scalars), -1)
        # X_train_micro = np.load(TENSOR_DIR / "train_micro.npy").reshape(len(X_train_scalars), -1)
        Y_train = np.load(TENSOR_DIR / "train_targets.npy")
        
        X_val_scalars = np.load(TENSOR_DIR / "val_scalars.npy")
        X_val_macro = np.load(TENSOR_DIR / "val_macro.npy").reshape(len(X_val_scalars), -1)
        # X_val_micro = np.load(TENSOR_DIR / "val_micro.npy").reshape(len(X_val_scalars), -1)
        Y_val = np.load(TENSOR_DIR / "val_targets.npy")
    except FileNotFoundError:
        logger.error("Tensors not found. Make sure tensor_builder_v3.py has finished running!")
        return

    # To save RAM and speed up XGBoost, we'll only load Micro for Model C if requested.
    # For now, let's just do A and B as they are extremely fast.
    
    # Feature sets
    X_train_A = X_train_scalars
    X_val_A = X_val_scalars
    
    X_train_B = np.hstack([X_train_scalars, X_train_macro])
    X_val_B = np.hstack([X_val_scalars, X_val_macro])
    
    # Target mapping
    # 0: y_reversion, 1: y_full_reversal, 2: strong_reversion
    targets = {
        "Reversion (Touch Open)": 0,
        "Full Reversal (Opposite Brick)": 1,
        "Strong Reversion (Depth > 1.5)": 2
    }
    
    xgb_params = {
        'n_estimators': 150,
        'max_depth': 4,
        'learning_rate': 0.05,
        'tree_method': 'hist',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    logger.info("\n================ MODEL A (Scalars Only) ================")
    for t_name, t_idx in targets.items():
        y_train = Y_train[:, t_idx]
        y_val = Y_val[:, t_idx]
        
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_train_A, y_train)
        evaluate_model(clf, X_val_A, y_val, t_name, "Model A")
        
    logger.info("\n================ MODEL B (Scalars + Macro) ================")
    for t_name, t_idx in targets.items():
        y_train = Y_train[:, t_idx]
        y_val = Y_val[:, t_idx]
        
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_train_B, y_train)
        evaluate_model(clf, X_val_B, y_val, t_name, "Model B")
        
    logger.info("\n================ MODEL C (Scalars + Macro + Micro) ================")
    logger.info("Loading Micro tensors (this may take a moment)...")
    X_train_micro = np.load(TENSOR_DIR / "train_micro.npy").reshape(len(X_train_scalars), -1)
    X_val_micro = np.load(TENSOR_DIR / "val_micro.npy").reshape(len(X_val_scalars), -1)
    
    X_train_C = np.hstack([X_train_scalars, X_train_macro, X_train_micro])
    X_val_C = np.hstack([X_val_scalars, X_val_macro, X_val_micro])
    
    for t_name, t_idx in targets.items():
        y_train = Y_train[:, t_idx]
        y_val = Y_val[:, t_idx]
        
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_train_C, y_train)
        evaluate_model(clf, X_val_C, y_val, t_name, "Model C")
        
    logger.info("\nExperiment Complete.")

if __name__ == "__main__":
    main()
