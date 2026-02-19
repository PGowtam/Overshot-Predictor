"""
Phase 8: Evaluation (FR-EV-01) - Robust Implementation

Evaluates model on Test Set (Jul-Dec 2023) and Holdout (2024).
Uses manual batching for prediction to avoid TF model.predict() hangs on Mac Metal.
"""

import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"
MODEL_PATH = OUTPUT_DIR / "model.keras"
CONFIG_PATH = OUTPUT_DIR / "config.json"
PLOT_DIR = OUTPUT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def load_tensors(split):
    print(f"📂 Loading {split} tensors...", flush=True)
    try:
        micro = np.load(TENSOR_DIR / f"{split}_micro.npy")
        macro = np.load(TENSOR_DIR / f"{split}_macro.npy")
        y_class = np.load(TENSOR_DIR / f"{split}_y_class.npy")
        y_mag = np.load(TENSOR_DIR / f"{split}_y_mag.npy")
        return micro, macro, y_class, y_mag
    except FileNotFoundError:
        print(f"⚠️  {split} tensors not found.", flush=True)
        return None, None, None, None

def safe_predict(model, micro, macro, batch_size=32):
    """Predict using manual batch loop to avoid model.predict() hangs."""
    n_samples = len(micro)
    prob_wins = []
    pred_oss = []
    
    print(f"🔮 Predicting {n_samples} samples (batch_size={batch_size})...", flush=True)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_micro = micro[i:end]
        batch_macro = macro[i:end]
        
        # Eager execution call
        preds = model([batch_micro, batch_macro], training=False)
        
        # Collect results
        prob_wins.append(preds[0].numpy().flatten())
        pred_oss.append(preds[1].numpy().flatten())
        
        if (i // batch_size) % 10 == 0:
             print(f"   Batch {i // batch_size} done...", flush=True)

    return np.concatenate(prob_wins), np.concatenate(pred_oss)

def evaluate():
    # 1. Load Config & Model
    if not CONFIG_PATH.exists():
        print("❌ Config not found.", flush=True)
        return
    config = load_config()
    th_prob = config["Prob_Win_threshold"]
    th_os = config["Pred_OS_threshold"]
    
    print(f"⚙️  Config Trigger: Prob_Win >= {th_prob}, Pred_OS >= {th_os}", flush=True)
    
    if not MODEL_PATH.exists():
        print("❌ Model not found.", flush=True)
        return
    print("🏗️  Loading model...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Evaluate function
    def run_eval(split_name):
        print("\n" + "="*40, flush=True)
        print(f" 🧪 EVALUATION: {split_name.upper()}", flush=True)
        print("="*40, flush=True)
        
        micro, macro, y_class, y_mag = load_tensors(split_name)
        if micro is None or len(micro) == 0:
            print(f"⚠️  Skipping {split_name} (No data).", flush=True)
            return

        # Predict
        try:
            prob_win, pred_os = safe_predict(model, micro, macro)
        except Exception as e:
            print(f"❌ Prediction failed for {split_name}: {e}", flush=True)
            return

        # Metrics
        # A. Unfiltered WR
        baseline_wr = np.mean(y_class)
        print(f"\nBaseline Win Rate (Unfiltered): {baseline_wr:.2%}", flush=True)
        
        # B. Model-Filtered WR
        mask = (prob_win >= th_prob) & (pred_os >= th_os)
        n_trades = np.sum(mask)
        
        if n_trades > 0:
            filtered_wins = y_class[mask]
            filtered_wr = np.mean(filtered_wins)
            print(f"Model-Filtered Win Rate:      {filtered_wr:.2%} ({n_trades} trades)", flush=True)
            
            target = 0.60 if split_name == "test" else 0.55
            if filtered_wr >= target:
                print(f"✅ TARGET MET (>= {target:.0%})", flush=True)
            else:
                print(f"❌ TARGET MISSED (< {target:.0%})", flush=True)
                
            # Plot Confusion Matrix (Trades Taken vs Outcome)
            # True Positive = Took trade & Won
            # False Positive = Took trade & Lost
            tp = np.sum(filtered_wins == 1)
            fp = np.sum(filtered_wins == 0)
            print(f"   Trades Taken Breakdown: {tp} Wins, {fp} Losses", flush=True)
            
        else:
            print("⚠️  No trades passed the filter!", flush=True)

        # C. Head B Pearson r on WIN samples
        win_mask = (y_class == 1)
        if np.sum(win_mask) > 0:
            actual_wins_mag = y_mag[win_mask]
            pred_wins_mag = pred_os[win_mask]
            
            corr = np.corrcoef(actual_wins_mag, pred_wins_mag)[0, 1]
            print(f"\nHead B (Pred_OS) Pearson r (on WINS): {corr:.4f}", flush=True)
            
            if split_name == "test":
                if corr >= 0.30:
                    print("✅ TARGET MET (>= 0.30)", flush=True)
                else:
                    print("❌ TARGET MISSED (< 0.30)", flush=True)
                
                # Plot Scatter
                plt.figure(figsize=(6, 6))
                plt.scatter(actual_wins_mag, pred_wins_mag, alpha=0.3, s=10)
                plt.xlabel("Actual Magnitude (y_mag)")
                plt.ylabel("Predicted Magnitude (Pred_OS)")
                plt.title(f"{split_name.title()} Correlation (Wins Only): r={corr:.3f}")
                plt.grid(True, alpha=0.3)
                plot_path = PLOT_DIR / f"{split_name}_correlation.png"
                plt.savefig(plot_path)
                print(f"   Scatter plot saved to {plot_path}", flush=True)
        
        # D. Pred_OS > 1.0 ratio
        # "WIN predictions" means model says WIN.
        model_win_preds_mask = (prob_win > 0.5)
        if np.sum(model_win_preds_mask) > 0:
            pred_os_on_wins = pred_os[model_win_preds_mask]
            ratio_gt_1 = np.mean(pred_os_on_wins > 1.0)
            print(f"\nPred_OS > 1.0 Ratio (on Prob_Win > 0.5): {ratio_gt_1:.2%}", flush=True)

    # Run for Test and Holdout
    run_eval("test")
    run_eval("holdout")

if __name__ == "__main__":
    evaluate()
