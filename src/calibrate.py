"""
Phase 7: Threshold Calibration (FR-CAL)

Calibrates Prob_Win and Pred_OS thresholds on the validation set.
Saves optimal thresholds to outputs/config.json.
Generates Plots:
  - outputs/plots/pr_curve.png
  - outputs/plots/pred_os_dist.png
"""

import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, precision_score
import matplotlib.pyplot as plt

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"
MODEL_PATH = OUTPUT_DIR / "model.keras"
PLOT_DIR = OUTPUT_DIR / "plots"
CONFIG_PATH = OUTPUT_DIR / "config.json"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_val_data():
    print("📂 Loading validation tensors...")
    micro = np.load(TENSOR_DIR / "val_micro.npy")
    macro = np.load(TENSOR_DIR / "val_macro.npy")
    y_class = np.load(TENSOR_DIR / "val_y_class.npy")
    y_mag = np.load(TENSOR_DIR / "val_y_mag.npy")
    return micro, macro, y_class, y_mag

def calibrate():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    # 1. Load Data/Model
    micro, macro, y_true_class, y_true_mag = load_val_data()
    print("🏗️  Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Predict
    print("🔮 Generating predictions...")
    preds = model.predict([micro, macro], verbose=1)
    prob_win = preds[0].flatten()
    pred_os = preds[1].flatten()
    
    # 3. Calibrate Head A (Prob_Win)
    print("\n📊 Calibrating Head A (Prob_Win)...")
    precisions, recalls, thresholds = precision_recall_curve(y_true_class, prob_win)
    
    # Plot PR Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, label='Head A')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Validation)')
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_DIR / "pr_curve.png")
    
    # Find threshold for Precision >= 0.60
    # Note: thresholds array is shorter than precisions/recalls by 1
    target_prec = 0.60
    selected_th_a = 0.5 # default
    
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p >= target_prec:
            selected_th_a = float(t)
            print(f"   found threshold {t:.4f} with precision {p:.4f} (recall {r:.4f})")
            break
    else:
        print("   WARNING: No threshold found with precision >= 0.60. Using max precision threshold.")
        idx = np.argmax(precisions[:-1])
        selected_th_a = float(thresholds[idx])

    # Ensure reasonable bounds
    selected_th_a = min(max(selected_th_a, 0.5), 0.95)
    print(f"✅ Selected Prob_Win_threshold: {selected_th_a:.4f}")

    # 4. Calibrate Head B (Pred_OS)
    print("\n📊 Calibrating Head B (Pred_OS)...")
    
    win_mask = (y_true_class == 1)
    loss_mask = (y_true_class == 0)
    
    pred_os_wins = pred_os[win_mask]
    pred_os_losses = pred_os[loss_mask]
    
    # Plot Distributions
    plt.figure(figsize=(10, 6))
    plt.hist(pred_os_wins, bins=50, alpha=0.5, label='Actual WINS', density=True, color='green')
    plt.hist(pred_os_losses, bins=50, alpha=0.5, label='Actual LOSSES', density=True, color='red')
    plt.xlabel('Pred_OS')
    plt.ylabel('Density')
    plt.title('Pred_OS Distribution by Outcome')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_DIR / "pred_os_dist.png")
    
    # Select threshold where WIN density > LOSS density significantly
    # Heuristic: Start at 1.1 as per PRD
    selected_th_b = 1.1
    print(f"✅ Selected Pred_OS_threshold: {selected_th_b:.4f} (PRD Default)")
    
    # 5. Save Config
    config = {
        "Prob_Win_threshold": round(selected_th_a, 4),
        "Pred_OS_threshold": round(selected_th_b, 4),
        "z_score_window": 1000,
        "micro_buffer_size": 100,
        "macro_history_size": 10
    }
    
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"\n💾 Config saved to {CONFIG_PATH}")
    print(json.dumps(config, indent=2))

if __name__ == "__main__":
    calibrate()
