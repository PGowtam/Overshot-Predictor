"""
Independent Analysis: Pred_OS Thresholds

Analyzes Pred_OS performance WITHOUT Prob_Win filter.
Checks if Pred_OS alone is sufficient for high-probability signals.
"""

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"
MODEL_PATH = OUTPUT_DIR / "model.keras"

def analyze_independent():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    # 1. Load Data
    print("📂 Loading validation data...")
    micro = np.load(TENSOR_DIR / "val_micro.npy")
    macro = np.load(TENSOR_DIR / "val_macro.npy")
    y_class = np.load(TENSOR_DIR / "val_y_class.npy")
    y_mag = np.load(TENSOR_DIR / "val_y_mag.npy")

    # 2. Predict
    print("🔮 Generating predictions...")
    model = tf.keras.models.load_model(MODEL_PATH)
    preds = model.predict([micro, macro], verbose=0)
    prob_win = preds[0].flatten()
    pred_os = preds[1].flatten()
    
    # 3. Independent Analysis (No Prob_Win filter)
    thresholds = np.linspace(0.5, 2.0, 31)
    
    print(f"\nIndependent Analysis (Prob_Win Ignored):")
    print(f"{'Pred_OS Th':<12} | {'Win Rate':<10} | {'Trades':<8} | {'Win Count':<8}")
    print("-" * 45)
    
    best_wr = 0
    best_th = 0
    
    for th in thresholds:
        combined_mask = (pred_os >= th)
        n_trades = np.sum(combined_mask)
        
        if n_trades > 0:
            wins = y_class[combined_mask]
            n_wins = np.sum(wins)
            wr = n_wins / n_trades
            
            if wr > best_wr and n_trades >= 50:
                best_wr = wr
                best_th = th
            
            print(f"{th:<12.2f} | {wr:<10.2%} | {n_trades:<8} | {n_wins:<8}")
        else:
            print(f"{th:<12.2f} | {'N/A':<10} | {0:<8} | {0:<8}")

    print(f"\n🏆 Best Threshold (Independent): {best_th:.2f} (WR: {best_wr:.2%})")

if __name__ == "__main__":
    analyze_independent()
