"""
Calibrate EV V2
===============
1. Loads the trained Transformer.
2. Evaluates on Validation Set.
3. Applies Temperature Scaling to the prob_win logits.
4. Applies Isotonic Regression to the pred_os output.
5. Computes the final EV equation and validates Top Decile edge.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU to prevent Mac GPU deadlock
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # Disable Metal/MPS
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from models_exec_v2 import BinaryFocalCrossentropy

BASE_DIR = Path(__file__).resolve().parent.parent
TENSOR_DIR = BASE_DIR / "outputs" / "exec_tensors_v2"
MODEL_DIR = BASE_DIR / "outputs" / "transformer_v2"
OUTPUT_DIR = BASE_DIR / "outputs" / "calibration_v2"

logger = logging.getLogger(__name__)

def load_data(split_name):
    logger.info(f"Loading {split_name} data...")
    micro = np.load(TENSOR_DIR / f"{split_name}_micro.npy")
    macro = np.load(TENSOR_DIR / f"{split_name}_macro.npy")
    summary = np.load(TENSOR_DIR / f"{split_name}_summary.npy")
    y_class = np.load(TENSOR_DIR / f"{split_name}_y_class.npy")
    y_mag = np.load(TENSOR_DIR / f"{split_name}_y_mag.npy")
    
    y_class = y_class.flatten()
    y_mag = y_mag.flatten()
    valid_mask = ~np.isnan(y_class) & ~np.isnan(y_mag)
    
    X = {
        'micro_input': micro[valid_mask],
        'macro_input': macro[valid_mask],
        'summary_input': summary[valid_mask]
    }
    return X, y_class[valid_mask], y_mag[valid_mask]

# Define custom objects for loading the model
custom_objects = {
    'BinaryFocalCrossentropy': BinaryFocalCrossentropy,
    'PositionEmbedding': tf.keras.layers.Layer, # these are saved with config
    'TransformerEncoderBlock': tf.keras.layers.Layer,
    'ResidualBlock': tf.keras.layers.Layer,
}

def logit(p):
    """Inverse sigmoid."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def optimize_temperature(probs, y_true):
    """Find temperature T that minimizes NLL."""
    logits = logit(probs)
    
    def nll(T):
        scaled_logits = logits / T
        scaled_probs = sigmoid(scaled_logits)
        eps = 1e-7
        p = np.clip(scaled_probs, eps, 1 - eps)
        # NLL
        loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        return loss
        
    res = minimize(nll, x0=[1.5], bounds=[(0.5, 5.0)])
    T = res.x[0]
    logger.info(f"Optimized Temperature T = {T:.4f}")
    return T

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / "calibration.log"), mode='w')
        ]
    )
    
    # 1. Load Model
    model_path = MODEL_DIR / "model_best.keras"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
        
    logger.info("Loading Transformer model...")
    # Safe load using custom objects map for custom loss/layers
    from tensorflow.keras.models import load_model
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    
    # 2. Load Validation Data
    X_val, y_val_class, y_val_reg = load_data("val")
    
    # 3. Predict (Manually chunked to avoid memory/predict deadlocks)
    logger.info("Running inference on validation set (CPU)...")
    raw_p_win_list = []
    raw_y_reg_list = []
    
    batch_size = 512
    n_samples = len(y_val_class)
    for i in range(0, n_samples, batch_size):
        batch_X = {k: v[i:i+batch_size] for k, v in X_val.items()}
        preds = model(batch_X, training=False)
        raw_p_win_list.append(preds[0].numpy().flatten())
        raw_y_reg_list.append(preds[1].numpy().flatten())
        
    raw_p_win = np.concatenate(raw_p_win_list)
    raw_y_reg = np.concatenate(raw_y_reg_list)
    
    # 4. Temperature Scaling (Classification)
    logger.info("Applying Temperature Scaling...")
    T = optimize_temperature(raw_p_win, y_val_class)
    calibrated_p_win = sigmoid(logit(raw_p_win) / T)
    
    # 5. Isotonic Regression (Regression)
    logger.info("Applying Isotonic Regression...")
    iso = IsotonicRegression(out_of_bounds='clip')
    calibrated_y_reg = iso.fit_transform(raw_y_reg, y_val_reg)
    
    # 6. Compute EV
    logger.info("Computing Enhanced Expected Value (EV)...")
    # Equation: EV = p_win_calibrated * y_reg_calibrated / (1 - p_win_calibrated)
    # (Leaving out p_strong as it was excluded in V1)
    ev_scores = (calibrated_p_win * calibrated_y_reg) / (1 - calibrated_p_win + 1e-8)
    
    # 7. Select Top Decile
    threshold_90 = np.percentile(ev_scores, 90)
    top_10_mask = ev_scores >= threshold_90
    
    selected_y_class = y_val_class[top_10_mask]
    selected_y_reg = y_val_reg[top_10_mask]
    
    win_rate = np.mean(selected_y_class)
    avg_overshoot = np.mean(selected_y_reg)
    
    wins = np.sum(selected_y_class == 1)
    losses = np.sum(selected_y_class == 0)
    profit_factor = wins / (losses + 1e-8)
    
    logger.info("========== CALIBRATION RESULTS ==========")
    logger.info(f"Top 10% EV Threshold: {threshold_90:.4f}")
    logger.info(f"Selected Samples: {len(selected_y_class)}")
    logger.info(f"Empirical Win Rate: {win_rate*100:.2f}% (Target: >= 55%)")
    logger.info(f"Average Realized Overshoot: {avg_overshoot:.4f} (Target: >= 1.5)")
    logger.info(f"Profit Factor: {profit_factor:.2f} ({wins}W / {losses}L)")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
