"""
Phase 9.3: Target Execution-Priced Tensors for Training (Option B)

Trains a brand new model from scratch using the execution-priced
labels and tensors generated in Phase 9.1.

Output:
  - outputs/exec/model.keras
  - outputs/exec/config.json
  - outputs/exec/training_log.csv
"""

import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
)

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from model import build_model, compile_model

# Paths
EXEC_DIR = BASE_DIR / "outputs" / "exec"
TENSOR_DIR = EXEC_DIR / "tensors"
MODEL_PATH = EXEC_DIR / "model.keras"
LOG_PATH = EXEC_DIR / "training_log.csv"
CONFIG_PATH = EXEC_DIR / "config.json"
PLOT_DIR = EXEC_DIR / "plots"

BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE_ES = 15
PATIENCE_LR = 8
FACTOR_LR = 0.5


class OverfittingMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch < 20: return
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if val_loss and train_loss and val_loss > 1.5 * train_loss:
            print(f"\n⚠️  WARNING: Potential overfitting at epoch {epoch+1}")


def load_tensors():
    """Load train/val tensors."""
    print("📂 Loading execution-priced tensors...")
    data = {}
    for split in ["train", "val"]:
        micro = np.load(TENSOR_DIR / f"{split}_micro.npy")
        macro = np.load(TENSOR_DIR / f"{split}_macro.npy")
        y_class = np.load(TENSOR_DIR / f"{split}_y_class.npy")
        y_mag = np.load(TENSOR_DIR / f"{split}_y_mag.npy")
        
        weights = None
        w_path = TENSOR_DIR / f"{split}_weights.npy"
        if w_path.exists():
            weights = np.load(w_path)
            
        data[split] = {
            "micro": micro, "macro": macro, "y_class": y_class, "y_mag": y_mag, "weights": weights
        }
        print(f"  {split}: {len(y_class):,} samples")
    return data


def calibrate_and_save_config():
    print("\n" + "=" * 60)
    print(" 9.3c — THRESHOLD CALIBRATION")
    print("=" * 60)
    
    micro = np.load(TENSOR_DIR / "val_micro.npy")
    macro = np.load(TENSOR_DIR / "val_macro.npy")
    y_class = np.load(TENSOR_DIR / "val_y_class.npy")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    preds = model.predict([micro, macro], verbose=0)
    prob_win = preds[0].flatten()
    pred_os = preds[1].flatten()
    
    # Grid search for best Pred_OS threshold with Prob_Win >= 0.5
    print("\n🔍 Scanning thresholds on validation set...")
    prob_th = 0.5
    prob_mask = (prob_win >= prob_th)
    
    best_wr = 0
    best_th = 1.0 # fallback
    
    os_thresholds = np.linspace(0.8, 2.0, 25)
    for th in os_thresholds:
        mask = prob_mask & (pred_os >= th)
        n_trades = np.sum(mask)
        if n_trades >= 50:
            wr = np.mean(y_class[mask])
            if wr > best_wr:
                best_wr = wr
                best_th = th
                
    # To be safe, try not to be overly greedy on val set if it drops trades too much.
    # The user configured 1.3 on mid-price. We'll pick the one that maximizes WR.
    print(f"🏆 Best Validation Pred_OS Threshold (min 50 trades): {best_th:.2f} (WR: {best_wr:.2%})")

    config = {
        "Prob_Win_threshold": float(prob_th),
        "Pred_OS_threshold": float(round(best_th, 2)),
        "z_score_window": 1000,
        "micro_buffer_size": 100,
        "macro_history_size": 10
    }
    
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"💾 Config saved to {CONFIG_PATH}")


def main():
    print("=" * 60)
    print(" Phase 9.3: Model Retraining (Option B)")
    print("=" * 60)
    
    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    data = load_tensors()
    
    print("\n🏗️  Building new model...")
    model = build_model()
    model = compile_model(model)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE_ES, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=FACTOR_LR, patience=PATIENCE_LR, verbose=1, min_lr=1e-6),
        ModelCheckpoint(filepath=str(MODEL_PATH), monitor='val_loss', save_best_only=True, verbose=1),
        CSVLogger(str(LOG_PATH)),
        OverfittingMonitor()
    ]
    
    print(f"\n🚀 Starting training (max epochs={MAX_EPOCHS}, batch={BATCH_SIZE})...")
    model.fit(
        x=[data["train"]["micro"], data["train"]["macro"]],
        y=[data["train"]["y_class"], data["train"]["y_mag"]],
        validation_data=(
            [data["val"]["micro"], data["val"]["macro"]],
            [data["val"]["y_class"], data["val"]["y_mag"]]
        ),
        sample_weight=[data["train"]["weights"], data["train"]["weights"]] if data["train"]["weights"] is not None else None,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Training complete.")
    calibrate_and_save_config()


if __name__ == "__main__":
    main()
