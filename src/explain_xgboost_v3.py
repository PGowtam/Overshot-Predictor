"""
Track B: XGBoost SHAP Explainability
====================================
Re-trains XGBoost Model A (Scalars only) for Reversion, and calculates SHAP values
to extract feature importance, interaction values, dependence plots, and optimized rule ranges.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
TENSOR_DIR = BASE_DIR / "outputs" / "exec_tensors_v3"
OUT_DIR = BASE_DIR / "outputs" / "experiments" / "shap_analysis"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = ['spread_current', 'ofi_peak', 'ofi_slope', 'wick_ratio', 'absorption_index', 'vel_peak', 'vel_current']

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading V3 Tensors...")
    
    try:
        X_train = np.load(TENSOR_DIR / "train_scalars.npy")
        Y_train = np.load(TENSOR_DIR / "train_targets.npy")
        
        X_val = np.load(TENSOR_DIR / "val_scalars.npy")
        Y_val = np.load(TENSOR_DIR / "val_targets.npy")
    except FileNotFoundError:
        logger.error("Tensors not found!")
        return

    # Target: 0 is Reversion (Touch Open)
    y_train = Y_train[:, 0]
    y_val = Y_val[:, 0]
    
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
    
    logger.info("Training Model A (Scalars) for SHAP Analysis...")
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    
    logger.info("Computing SHAP values...")
    X_val_df = pd.DataFrame(X_val, columns=FEATURE_NAMES)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val_df)
    
    # 1. Global Feature Importance
    logger.info("\n=== A. Global Feature Importance (Mean |SHAP|) ===")
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feat_imp = pd.DataFrame({'Feature': FEATURE_NAMES, 'Mean_SHAP': mean_abs_shap})
    feat_imp = feat_imp.sort_values('Mean_SHAP', ascending=False)
    for _, row in feat_imp.iterrows():
        logger.info(f"{row['Feature']:<20} {row['Mean_SHAP']:.4f}")
        
    # 2. SHAP Dependence Plots
    logger.info("\n=== Generating SHAP Dependence Plots ===")
    for feat in ['spread_current', 'ofi_peak', 'wick_ratio']:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat, shap_values, X_val_df, show=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"shap_dependence_{feat}.png")
        plt.close()
        logger.info(f"Saved {feat} dependence plot to {OUT_DIR}")
        
    # 3. SHAP Interactions
    logger.info("\n=== C. SHAP Interaction Values ===")
    try:
        interaction_values = explainer.shap_interaction_values(X_val_df)
        
        interactions = []
        for i in range(len(FEATURE_NAMES)):
            for j in range(i+1, len(FEATURE_NAMES)):
                mean_interact = np.mean(np.abs(interaction_values[:, i, j])) * 2
                interactions.append({
                    'Feature_A': FEATURE_NAMES[i],
                    'Feature_B': FEATURE_NAMES[j],
                    'Interaction': mean_interact
                })
        int_df = pd.DataFrame(interactions).sort_values('Interaction', ascending=False).head(10)
        logger.info("Top 10 Feature Interactions:")
        for _, row in int_df.iterrows():
            logger.info(f"{row['Feature_A']:<18} x {row['Feature_B']:<18} : {row['Interaction']:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute SHAP interactions: {e}")

    # 4. Top 20 SHAP Examples
    logger.info("\n=== B. Top 20 Reversion Examples ===")
    preds = model.predict_proba(X_val_df)[:, 1]
    top_20_idx = np.argsort(preds)[-20:][::-1]
    
    for rank, idx in enumerate(top_20_idx):
        logger.info(f"\nExample #{rank+1} | Base Value: {explainer.expected_value:.4f} | Prediction Prob: {preds[idx]:.4f} | Target: {y_val[idx]}")
        sv = shap_values[idx]
        top_feats = np.argsort(np.abs(sv))[-3:][::-1]
        for f_idx in top_feats:
            logger.info(f"  {FEATURE_NAMES[f_idx]:<18}: {X_val_df.iloc[idx, f_idx]:>8.4f} -> {sv[f_idx]:+.4f} SHAP")

    # 5. Rule Extraction (Top 1% SHAP Scores)
    logger.info("\n=== D. Rule Extraction (Top 1% SHAP regions) ===")
    
    for feat in ['spread_current', 'ofi_peak', 'wick_ratio', 'vel_peak']:
        f_idx = FEATURE_NAMES.index(feat)
        f_shap = shap_values[:, f_idx]
        
        threshold_99 = np.percentile(f_shap, 99)
        top_1_mask = f_shap >= threshold_99
        
        if np.sum(top_1_mask) > 0:
            feat_vals = X_val_df.loc[top_1_mask, feat]
            min_v, max_v = feat_vals.min(), feat_vals.max()
            mean_v = feat_vals.mean()
            logger.info(f"{feat:<18} Top 1% SHAP Region -> Range: [{min_v:7.4f}, {max_v:7.4f}], Mean: {mean_v:7.4f}")

if __name__ == "__main__":
    main()
