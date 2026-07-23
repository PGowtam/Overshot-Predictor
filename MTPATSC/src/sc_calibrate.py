import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Calibrator")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "setup_classifier"
TENSOR_DIR = OUTPUT_DIR / "tensors"

def main():
    val_path = TENSOR_DIR / "val.npz"
    model_path = OUTPUT_DIR / "model_inference.keras"
    
    if not val_path.exists() or not model_path.exists():
        logger.error("Required validation tensors or inference model not found. Run sc_train.py first.")
        return
        
    logger.info("Loading validation data and inference model...")
    val_data = np.load(val_path)
    model = tf.keras.models.load_model(str(model_path))
    
    # 1. Generate predictions
    x_val = [
        val_data['ancs_fine'],
        val_data['ancs_coarse'],
        val_data['history'],
        val_data['scalars']
    ]
    
    probs = model.predict(x_val) # Shape: (N, 5)
    pred_classes = np.argmax(probs, axis=1)
    
    # Extract binary win flags for WR calculations
    win_flags = {
        1: val_data['t1_win'],
        2: val_data['t2_win'],
        3: val_data['t3_win'],
        4: val_data['t4_win']
    }
    
    # Risk Reward profiles
    rr_profiles = {1: 1.0, 2: 2.0, 3: 2.0, 4: 3.0}
    
    # 2. Perform Threshold Sweeps
    optimized_thresholds = {}
    veto_threshold = 0.40
    
    logger.info("Starting per-class calibration sweep...")
    for setup_class in [1, 2, 3, 4]:
        rr = rr_profiles[setup_class]
        t_win = win_flags[setup_class]
        
        best_score = -1e9
        best_theta = 1.0 # Default to disabled
        best_metrics = {"n_trades": 0, "win_rate": 0.0, "ev": 0.0}
        
        # Sweep theta from 0.20 to 0.95 with fine steps of 0.01
        for theta in np.arange(0.20, 0.96, 0.01):
            # Apply confidence, prediction class, and T0 veto constraints
            mask = (pred_classes == setup_class) & (probs[:, setup_class] >= theta) & (probs[:, 0] <= veto_threshold)
            n_trades = int(mask.sum())
            
            if n_trades < 10:
                continue
                
            win_rate = float(np.mean(t_win[mask]))
            ev = win_rate * rr - (1.0 - win_rate) * 1.0
            score = ev * np.sqrt(n_trades)
            
            if ev > 0.0 and score > best_score:
                best_score = score
                best_theta = float(theta)
                best_metrics = {
                    "n_trades": n_trades,
                    "win_rate": round(win_rate, 4),
                    "ev": round(ev, 4)
                }
                
        optimized_thresholds[f"T{setup_class}_threshold"] = best_theta
        logger.info(f"T{setup_class} optimal threshold: {best_theta:.2f} (Trades: {best_metrics['n_trades']}, WR: {best_metrics['win_rate']:.2%}, EV: {best_metrics['ev']:.2f} R)")
        
    optimized_thresholds["T0_veto_threshold"] = veto_threshold
    
    # 3. Export Config
    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(optimized_thresholds, f, indent=4)
    logger.info(f"Saved calibrated thresholds to {config_path}")

if __name__ == "__main__":
    main()
