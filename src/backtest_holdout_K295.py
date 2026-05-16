import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

TENSOR_DIR = BASE_DIR / "outputs" / "tensors_holdout_K295"
MODEL_BASE_DIR = BASE_DIR / "BrickOfTicks_Trader" / "models"
FOLDS = ["fold_1", "fold_2", "fold_3"]

def load_holdout_tensors():
    print(f"📂 Loading holdout K295 tensors from {TENSOR_DIR}...")
    micro = np.load(TENSOR_DIR / "holdout_micro.npy")
    macro = np.load(TENSOR_DIR / "holdout_macro.npy")
    y_class = np.load(TENSOR_DIR / "holdout_y_class.npy")
    y_mag = np.load(TENSOR_DIR / "holdout_y_mag.npy")
    return micro, macro, y_class, y_mag

def safe_predict(model, micro, macro, batch_size=32):
    n_samples = len(micro)
    prob_wins = []
    pred_oss = []
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_micro = micro[i:end]
        batch_macro = macro[i:end]
        
        preds = model([batch_micro, batch_macro], training=False)
        prob_wins.append(preds[0].numpy().flatten())
        pred_oss.append(preds[1].numpy().flatten())

    return np.concatenate(prob_wins), np.concatenate(pred_oss)

def main():
    # 1. Load Data
    micro, macro, y_class, y_mag = load_holdout_tensors()
    n_samples = len(micro)
    print(f"✅ Loaded {n_samples} samples.")

    # 2. Backtest Models
    
    # --- A. Single "Exec" Model (outputs/model.keras) ---
    print("\n--- Testing Single Exec Model (Main) ---")
    model_main_path = BASE_DIR / "outputs" / "model.keras"
    model_main = tf.keras.models.load_model(model_main_path)
    main_prob_win, main_pred_os = safe_predict(model_main, micro, macro)
    
    # --- B. 3-Fold Ensemble Models ---
    print("\n--- Testing 3-Fold Ensemble Models ---")
    all_prob_wins = []
    all_pred_oss = []

    for fold in FOLDS:
        model_path = MODEL_BASE_DIR / fold
        print(f"🏗️  Loading {fold}...")
        if (model_path / "model.keras").exists():
            model = tf.keras.models.load_model(model_path / "model.keras")
        else:
            model = tf.keras.models.load_model(model_path)
            
        pw, po = safe_predict(model, micro, macro)
        all_prob_wins.append(pw)
        all_pred_oss.append(po)

    avg_prob_win = np.mean(all_prob_wins, axis=0)
    avg_pred_os = np.mean(all_pred_oss, axis=0)

    # 3. Report Generation
    print("\n" + "="*60)
    print(" FULL FEASIBILITY REPORT: HOLD-OUT K=0.00295")
    print("="*60)
    
    baseline_wr = np.mean(y_class)
    print(f"Naive Continuation Win Rate (1:1): {baseline_wr:.2%}")
    print(f"Structural Spread Burden:          5.9% (at K=0.00295)")
    
    thresholds = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    def calc_stats(p_win, p_os, label):
        print(f"\n[{label} Results]")
        res_list = []
        for th in thresholds:
            mask = (p_os >= th)
            n_trades = np.sum(mask)
            if n_trades > 0:
                wr = np.mean(y_class[mask])
                # Expectancy accounting for 5.9% spread cost
                exp = wr * 0.941 - (1-wr) * 1.059
                res_list.append({"Th": th, "Trades": n_trades, "WR": f"{wr:.2%}", "Exp": f"{exp:.3f}"})
        
        if res_list:
            print(pd.DataFrame(res_list).to_string(index=False))
        else:
            print("No trades triggered.")

    calc_stats(main_prob_win, main_pred_os, "Single Exec Model")
    calc_stats(avg_prob_win, avg_pred_os, "3-Fold CV Ensemble")

    print("\n" + "="*60)
    print(" FINAL VERDICT")
    print("="*60)
    # Check if either has positive expectancy at common threshold
    best_main_exp = max([float(r["Exp"]) for r in []]) # placeholder logic
    # I'll just print a textual conclusion based on the output.

if __name__ == "__main__":
    main()
