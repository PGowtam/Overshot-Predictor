"""
Volume Fallback: Threshold Calibration
=======================================
Calibrates Prob_Win and Pred_OS thresholds on the fallback validation set.

Same calibration logic as calibrate.py but uses fallback model and tensors.

Reads from:  outputs/fallback/tensors/, outputs/fallback/model.keras
Saves to:    outputs/fallback/config.json
             outputs/fallback/plots/
"""

import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

FALLBACK_DIR = BASE_DIR / "outputs" / "fallback"
TENSOR_DIR = FALLBACK_DIR / "tensors"
MODEL_PATH = FALLBACK_DIR / "model.keras"
PLOT_DIR = FALLBACK_DIR / "plots"
CONFIG_PATH = FALLBACK_DIR / "config.json"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


def safe_predict(model, micro, macro, batch_size=32):
    """Predict using manual batch loop to avoid model.predict() hangs on Mac Metal."""
    n_samples = len(micro)
    prob_wins = []
    pred_oss = []

    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        preds = model([micro[i:end], macro[i:end]], training=False)
        prob_wins.append(preds[0].numpy().flatten())
        pred_oss.append(preds[1].numpy().flatten())

    return np.concatenate(prob_wins), np.concatenate(pred_oss)


def calibrate():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    # 1. Load Data/Model
    print("📂 Loading fallback validation tensors...")
    micro = np.load(TENSOR_DIR / "val_micro.npy")
    macro = np.load(TENSOR_DIR / "val_macro.npy")
    y_true_class = np.load(TENSOR_DIR / "val_y_class.npy")
    y_true_mag = np.load(TENSOR_DIR / "val_y_mag.npy")

    print("🏗️  Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Predict
    print("🔮 Generating predictions...")
    prob_win, pred_os = safe_predict(model, micro, macro)

    # 3. Calibrate Head A (Prob_Win)
    print("\n📊 Calibrating Head A (Prob_Win)...")
    precisions, recalls, thresholds = precision_recall_curve(y_true_class, prob_win)

    # Plot PR Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, label='Head A (Fallback)', color='#e67e22')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve — Fallback Model (Validation)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOT_DIR / "pr_curve.png", dpi=150)
    plt.close()

    # Find threshold for Precision >= 0.60
    target_prec = 0.60
    selected_th_a = 0.5  # default

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
    plt.title('Pred_OS Distribution — Fallback Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_DIR / "pred_os_dist.png", dpi=150)
    plt.close()

    # Use PRD default
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
    print("=" * 60)
    print(" Volume Fallback: Threshold Calibration")
    print("=" * 60)
    calibrate()
