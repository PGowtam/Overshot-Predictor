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
    fixed_prob_th = 0.5
    prob_mask = (prob_win >= fixed_prob_th)
    
    thresholds = np.linspace(0.5, 2.0, 31) # 0.5 to 2.0 in 0.05 steps
    win_rates = []
    counts = []
    
    print(f"\nSensitivity Analysis (Prob_Win >= {fixed_prob_th}):")
    print(f"{'Pred_OS Th':<12} | {'Win Rate':<10} | {'Trades':<8} | {'Win Count':<8}")
    print("-" * 45)
    
    best_wr = 0
    best_th = 0
    
    for th in thresholds:
        combined_mask = prob_mask & (pred_os >= th)
        n_trades = np.sum(combined_mask)
        
        if n_trades > 0:
            # Win defined as y_mag >= 1.0 (or y_class == 1, should match)
            # Using y_class for ground truth win
            wins = y_class[combined_mask]
            n_wins = np.sum(wins)
            wr = n_wins / n_trades
            win_rates.append(wr)
            counts.append(n_trades)
            
            if wr > best_wr and n_trades >= 50: # Min sample constraint
                best_wr = wr
                best_th = th
            
            print(f"{th:<12.2f} | {wr:<10.2%} | {n_trades:<8} | {n_wins:<8}")
        else:
            win_rates.append(0)
            counts.append(0)
            print(f"{th:<12.2f} | {'N/A':<10} | {0:<8} | {0:<8}")

    # 4. Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Pred_OS Threshold')
    ax1.set_ylabel('Win Rate', color=color)
    ax1.plot(thresholds, win_rates, color=color, marker='o', label='Win Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.0) # Zoom in on relevant range

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('Trade Count', color=color)  # we already handled the x-label with ax1
    ax2.plot(thresholds, counts, color=color, linestyle='--', label='Count')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Win Rate vs Pred_OS Threshold (Prob_Win >= {fixed_prob_th})')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(PLOT_DIR / "threshold_sensitivity.png")
    print(f"\n📊 Sensitivity plot saved to {PLOT_DIR / 'threshold_sensitivity.png'}")
    
    print(f"\n🏆 Best Threshold (min 50 trades): {best_th:.2f} (WR: {best_wr:.2%})")

if __name__ == "__main__":
    analyze_sensitivity()
