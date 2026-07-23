import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRADES_PATH = BASE_DIR / "outputs" / "experiments" / "rule_test_2026_v4" / "rule_test_2026_v4_trades.parquet"
MODEL_PATH = BASE_DIR / "outputs" / "xgboost_v4_pct" / "xgboost_v4_pct.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading 2026 BlackBull Trades...")
    df = pd.read_parquet(TRADES_PATH)
    
    # Filter to Rule A and resolve FORCE_CLOSE
    ra = df[(df['strategy'] == 'Rule A') & (df['outcome'] != 'FORCE_CLOSE')].copy()
    ra = ra.sort_values('entry_time').reset_index(drop=True)
    ra['win'] = (ra['outcome'] == 'WIN').astype(int)
    
    # Apply Execution Filter (Deduplicate millisecond gap-fills)
    exact_dupes = ra.duplicated(subset=['entry_time'], keep='first')
    clean_trades = ra[~exact_dupes].copy()
    
    logger.info(f"Raw Trades (95/20): {len(ra)}")
    logger.info(f"Gap-fill duplicates removed: {exact_dupes.sum()}")
    logger.info(f"Valid Executable Trades: {len(clean_trades)}")
    
    # Hardcoded Baseline (99/3)
    baseline_mask = (clean_trades['spread_pct'] >= 99) & (clean_trades['ofi_pct'] <= 3)
    baseline = clean_trades[baseline_mask]
    b_t = len(baseline)
    b_w = baseline['win'].sum()
    b_hr = b_w / b_t if b_t > 0 else 0
    b_pf = baseline[baseline['pnl_r']>0]['pnl_r'].sum() / (abs(baseline[baseline['pnl_r']<0]['pnl_r'].sum()) + 1e-8)
    
    logger.info(f"\n--- HARDCODED BASELINE (99/3) ---")
    logger.info(f"Trades: {b_t}")
    logger.info(f"Win Rate: {b_hr*100:.2f}%")
    logger.info(f"Profit Factor: {b_pf:.2f}")
    
    # Load Model
    logger.info("\nLoading XGBoost Model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Prepare features: ['spread_current_pct', 'abs_ofi_peak_pct', 'hour', 'weekday']
    days = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    clean_trades['dayofweek'] = clean_trades['weekday'].map(days)
    
    X = clean_trades[['spread_pct', 'ofi_pct', 'hour', 'dayofweek']].copy()
    X.columns = ['spread_current_pct', 'abs_ofi_peak_pct', 'hour', 'weekday']
    
    logger.info("Running ML Predictions on BlackBull data...")
    probs = model.predict_proba(X)[:, 1]
    clean_trades['ml_prob'] = probs
    
    logger.info(f"\n--- XGBOOST SMART FILTER ---")
    for prob_thresh in np.linspace(0.45, 0.65, 21):
        ml_mask = clean_trades['ml_prob'] >= prob_thresh
        ml_t = ml_mask.sum()
        if ml_t < 10:
            continue
            
        subset = clean_trades[ml_mask]
        ml_w = subset['win'].sum()
        ml_hr = ml_w / ml_t
        ml_gp = subset[subset['pnl_r']>0]['pnl_r'].sum()
        ml_gl = abs(subset[subset['pnl_r']<0]['pnl_r'].sum())
        ml_pf = ml_gp / (ml_gl + 1e-8)
        
        mark = "⭐⭐⭐" if ml_pf > 1.67 and ml_t > 50 else ""
        logger.info(f"Prob >= {prob_thresh:.2f} | Trades: {ml_t:3d} | HR: {ml_hr*100:.2f}% | PF: {ml_pf:.2f} {mark}")

if __name__ == "__main__":
    main()
