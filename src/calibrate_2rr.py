"""
2RR Overshot: Threshold Calibration
=====================================
Calibrates Prob_2RR and Pred_OS thresholds on the 2RR validation set.

Key difference: We target P(2-brick continuation) and need WR >= 33.3%
for 2:1 RR breakeven, not 50%.

Reads from:  outputs/fallback_2rr/{tensors/, model.keras}
Saves to:    outputs/fallback_2rr/config.json
             outputs/fallback_2rr/plots/
"""

import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

RR2_DIR = BASE_DIR / "outputs" / "fallback_2rr"
TENSOR_DIR = RR2_DIR / "tensors"
MODEL_PATH = RR2_DIR / "model.keras"
PLOT_DIR = RR2_DIR / "plots"
CONFIG_PATH = RR2_DIR / "config.json"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# For 2:1 RR, breakeven WR = 33.3%
BREAKEVEN_WR = 1.0 / 3.0


def safe_predict(model, micro, macro, batch_size=32):
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
        print(f"❌ 2RR Model not found at {MODEL_PATH}")
        return

    # 1. Load Data/Model
    print("📂 Loading 2RR validation tensors...")
    micro = np.load(TENSOR_DIR / "val_micro.npy")
    macro = np.load(TENSOR_DIR / "val_macro.npy")
    y_true_class = np.load(TENSOR_DIR / "val_y_class.npy")  # 2RR labels
    y_true_mag = np.load(TENSOR_DIR / "val_y_mag.npy")

    print("🏗️  Loading 2RR model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Predict
    print("🔮 Generating predictions...")
    prob_2rr, pred_os = safe_predict(model, micro, macro)

    # 3. ROC AUC
    try:
        auc = roc_auc_score(y_true_class, prob_2rr)
        print(f"\n📊 ROC AUC for 2RR prediction: {auc:.4f}")
    except Exception:
        auc = None
        print("\n⚠️  Could not compute AUC")

    # 4. Calibrate — Sweep thresholds
    print("\n📊 Threshold sweep (Head A: Prob_2RR)...")
    print(f"{'Threshold':>10s}  {'Trades':>8s}  {'2RR Wins':>10s}  {'2RR WR':>8s}  "
          f"{'EV (2:1)':>10s}  {'Viable':>8s}")
    print("-" * 70)

    best_ev = -999
    best_th = 0.5
    best_trades = 0

    for th in np.arange(0.30, 0.96, 0.05):
        mask = prob_2rr >= th
        n_trades = mask.sum()
        if n_trades < 5:
            continue
        wr = y_true_class[mask].mean()
        ev = 3 * wr - 1  # EV for 2:1 RR
        viable = "✅" if ev > 0 else "❌"
        print(f"{th:>10.2f}  {n_trades:>8d}  {int(y_true_class[mask].sum()):>10d}  "
              f"{wr:>8.1%}  {ev:>10.3f}R  {viable:>8s}")

        if ev > best_ev and n_trades >= 10:
            best_ev = ev
            best_th = float(th)
            best_trades = n_trades

    # Also sweep with Pred_OS filter
    print(f"\n📊 Combined threshold sweep (Prob_2RR + Pred_OS)...")
    print(f"{'Prob ≥':>8s}  {'OS ≥':>8s}  {'Trades':>8s}  {'2RR WR':>8s}  "
          f"{'EV':>8s}  {'Viable':>8s}")
    print("-" * 60)

    best_combo_ev = -999
    best_combo = (0.5, 1.0)

    for th_prob in np.arange(0.30, 0.91, 0.10):
        for th_os in [0.5, 1.0, 1.5, 2.0, 2.5]:
            mask = (prob_2rr >= th_prob) & (pred_os >= th_os)
            n_trades = mask.sum()
            if n_trades < 5:
                continue
            wr = y_true_class[mask].mean()
            ev = 3 * wr - 1
            viable = "✅" if ev > 0 else "❌"
            print(f"{th_prob:>8.2f}  {th_os:>8.1f}  {n_trades:>8d}  {wr:>8.1%}  "
                  f"{ev:>8.3f}R  {viable:>8s}")

            if ev > best_combo_ev and n_trades >= 10:
                best_combo_ev = ev
                best_combo = (float(th_prob), float(th_os))

    # Select best thresholds
    if best_combo_ev > best_ev:
        selected_th_prob, selected_th_os = best_combo
        print(f"\n✅ Best combo: Prob_2RR >= {selected_th_prob:.2f}, "
              f"Pred_OS >= {selected_th_os:.1f}  (EV={best_combo_ev:+.3f}R)")
    else:
        selected_th_prob = best_th
        selected_th_os = 1.0  # default
        print(f"\n✅ Best single: Prob_2RR >= {selected_th_prob:.2f}  (EV={best_ev:+.3f}R)")

    # 5. Plots
    # PR Curve
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true_class, prob_2rr)
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, color='#e74c3c', linewidth=2)
    plt.axhline(y=BREAKEVEN_WR, color='gray', linestyle='--', label=f'Breakeven ({BREAKEVEN_WR:.1%})')
    plt.xlabel('Recall')
    plt.ylabel('Precision (= Win Rate)')
    plt.title('Precision-Recall Curve — 2RR Model (Validation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_DIR / "pr_curve_2rr.png", dpi=150)
    plt.close()

    # Probability histogram
    plt.figure(figsize=(10, 6))
    plt.hist(prob_2rr[y_true_class == 1], bins=50, alpha=0.5, label='Actual 2RR WINs',
             density=True, color='green')
    plt.hist(prob_2rr[y_true_class == 0], bins=50, alpha=0.5, label='Actual 2RR LOSSes',
             density=True, color='red')
    plt.axvline(x=selected_th_prob, color='blue', linestyle='--',
                label=f'Threshold ({selected_th_prob:.2f})')
    plt.xlabel('Predicted P(2RR)')
    plt.ylabel('Density')
    plt.title('Model Probability Distribution — 2RR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_DIR / "prob_dist_2rr.png", dpi=150)
    plt.close()

    # Pred_OS distribution
    plt.figure(figsize=(10, 6))
    plt.hist(pred_os[y_true_class == 1], bins=50, alpha=0.5, label='Actual 2RR WINs',
             density=True, color='green')
    plt.hist(pred_os[y_true_class == 0], bins=50, alpha=0.5, label='Actual 2RR LOSSes',
             density=True, color='red')
    plt.xlabel('Pred_OS')
    plt.ylabel('Density')
    plt.title('Pred_OS Distribution — 2RR Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_DIR / "pred_os_dist_2rr.png", dpi=150)
    plt.close()

    # 6. Save Config
    config = {
        "Prob_2RR_threshold": round(selected_th_prob, 4),
        "Pred_OS_threshold": round(selected_th_os, 4),
        "target": "2RR (y_mag >= 2.0)",
        "breakeven_wr": round(BREAKEVEN_WR, 4),
        "roc_auc": round(auc, 4) if auc else None,
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
    print(" 2RR Overshot: Threshold Calibration")
    print("=" * 60)
    calibrate()
