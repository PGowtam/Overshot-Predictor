import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "xgboost_v4_pct"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    parquet_path = BASE_DIR / "outputs" / "sim_labels_v4" / "v4_percentiles_labels.parquet"
    logger.info(f"Loading V4 labels from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Drop 2020 (COVID anomaly)
    df = df[df['year'] > 2020].copy()
    
    # Extract time features
    dt = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['hour'] = dt.dt.hour
    df['weekday'] = dt.dt.dayofweek
    
    features = [
        'spread_current_pct',
        'abs_ofi_peak_pct',
        'hour',
        'weekday'
    ]
    target = 'reversion'
    
    # Train / Val Split
    train_mask = df['year'] <= 2025
    test_mask = df['year'] == 2026
    
    X_train = df[train_mask][features]
    y_train = df[train_mask][target]
    
    X_test = df[test_mask][features]
    y_test = df[test_mask][target]
    
    logger.info(f"Train size (2021-2025): {len(X_train)}")
    logger.info(f"Test size (2026): {len(X_test)}")
    
    logger.info(f"Class balance (Train): {y_train.mean():.4f}")
    
    # Calculate scale_pos_weight
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    scale_pos_weight = neg / pos
    
    logger.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='auc'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )
    
    # Save model
    model_path = OUTPUT_DIR / "xgboost_v4_pct.json"
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Evaluate on 2026 Test set (Dukascopy)
    logger.info("=== 2026 TEST EVALUATION ===")
    test_probs = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, test_probs)
    logger.info(f"ROC AUC: {auc:.4f}")
    
    # Baseline comparison (99/3 and 95/20 rules on Dukascopy)
    baseline_95_mask = (df[test_mask]['spread_current_pct'] >= 95) & (df[test_mask]['abs_ofi_peak_pct'] <= 20)
    baseline_95_trades = baseline_95_mask.sum()
    if baseline_95_trades > 0:
        baseline_hr = y_test[baseline_95_mask].mean()
        baseline_pf = y_test[baseline_95_mask].sum() / ((1 - y_test[baseline_95_mask]).sum() + 1e-8)
        logger.info(f"Baseline (95/20) - Trades: {baseline_95_trades}, HR: {baseline_hr*100:.2f}%, PF: {baseline_pf:.2f}")

    baseline_99_mask = (df[test_mask]['spread_current_pct'] >= 99) & (df[test_mask]['abs_ofi_peak_pct'] <= 3)
    baseline_99_trades = baseline_99_mask.sum()
    if baseline_99_trades > 0:
        baseline_hr = y_test[baseline_99_mask].mean()
        baseline_pf = y_test[baseline_99_mask].sum() / ((1 - y_test[baseline_99_mask]).sum() + 1e-8)
        logger.info(f"Baseline (99/3)  - Trades: {baseline_99_trades}, HR: {baseline_hr*100:.2f}%, PF: {baseline_pf:.2f}")
    
    # ML Threshold comparisons
    for pct in [90, 95, 98, 99]:
        threshold = np.percentile(test_probs, pct)
        ml_mask = test_probs >= threshold
        ml_trades = ml_mask.sum()
        if ml_trades > 0:
            ml_hr = y_test[ml_mask].mean()
            ml_pf = y_test[ml_mask].sum() / ((1 - y_test[ml_mask]).sum() + 1e-8)
            logger.info(f"ML Top {100-pct}% (Prob >= {threshold:.4f}) - Trades: {ml_trades}, HR: {ml_hr*100:.2f}%, PF: {ml_pf:.2f}")

if __name__ == "__main__":
    main()
