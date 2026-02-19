"""
Supplemental Analysis: Pred_OS Threshold Sensitivity

Analyzes the impact of Pred_OS_threshold on Win Rate and Trade Volume.
Assumes Prob_Win_threshold is fixed at 0.5.
Generates:
  - outputs/plots/threshold_sensitivity.png
  - Console report of Win Rate at various thresholds.
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
PLOT_DIR = OUTPUT_DIR / "plots"

def analyze_sensitivity():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    # 1. Load Data
    print("📂 Loading validation data...")
    micro = np.load(TENSOR_DIR / "val_micro.npy")
    macro = np.load(TENSOR_DIR / "val_macro.npy")
    y_class = np.load(TENSOR_DIR / "val_y_class.npy")
    y_mag = np.load(TENSOR_DIR / "val_y_mag.npy") # Actual magnitude
    

    # 2. Predict
    print("🔮 Generating predictions...")
    model = tf.keras.models.load_model(MODEL_PATH)
    preds = model.predict([micro, macro], verbose=0)
    prob_win = preds[0].flatten()
    pred_os = preds[1].flatten()
    
    # 3. Sensitivity Analysis
    scenarios = [
        ("Conditional (Prob_Win >= 0.7)", 0.7),
        ("Conditional (Prob_Win >= 0.8)", 0.8)
    ]
    
    thresholds = np.linspace(0.5, 2.0, 31)

    for name, min_prob in scenarios:
        prob_mask = (prob_win >= min_prob)
        print(f"\nAnalysis: {name}")
        print(f"{'Pred_OS Th':<12} | {'Win Rate':<10} | {'Trades':<8} | {'Win Count':<8}")
        print("-" * 45)
        
        best_wr = 0
        best_th = 0
        
        counts_for_plot = []
        wrs_for_plot = []

        for th in thresholds:
            combined_mask = prob_mask & (pred_os >= th)
            n_trades = np.sum(combined_mask)
            
            if n_trades > 0:
                wins = y_class[combined_mask]
                n_wins = np.sum(wins)
                wr = n_wins / n_trades
                wrs_for_plot.append(wr)
                counts_for_plot.append(n_trades)
                
                if wr > best_wr and n_trades >= 50:
                    best_wr = wr
                    best_th = th
                
                print(f"{th:<12.2f} | {wr:<10.2%} | {n_trades:<8} | {n_wins:<8}")
            else:
                wrs_for_plot.append(0)
                counts_for_plot.append(0)
                print(f"{th:<12.2f} | {'N/A':<10} | {0:<8} | {0:<8}")

        print(f"\n🏆 Best Threshold (min 50 trades): {best_th:.2f} (WR: {best_wr:.2%})")

    # 4. Plot (Only for fitst scenario to avoid clutter?)
    # ...


if __name__ == "__main__":
    analyze_sensitivity()
