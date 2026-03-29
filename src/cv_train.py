"""
Iteration 2: Cross Validation Training Pipeline

Loops over the 3 folds created by cv_tensor_builder.py.
For each fold:
1. Loads Train/Val tensors.
2. Initializes a new Model weights.
3. Trains the model (Early Stopping + ReduceLR identical to Phase 9)
4. Saves to outputs/exec/cv/fold_{1,2,3}/model.keras
5. Dynamically calibrates Thresholds (Prob_Win & Pred_OS) on the fold's Val set.
6. Saves config to config.json
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

EXEC_DIR = BASE_DIR / "outputs" / "exec"
CV_DIR = EXEC_DIR / "cv"

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


def load_fold_tensors(fold_num):
    print(f"\n📂 Loading Fold {fold_num} tensors...")
    tensor_dir = CV_DIR / f"fold_{fold_num}" / "tensors"
    
    if not (tensor_dir / "train_micro.npy").exists():
        raise FileNotFoundError(f"Fold {fold_num} tensors not found at {tensor_dir}")
        
    data = {}
    for split in ["train", "val"]:
        micro = np.load(tensor_dir / f"{split}_micro.npy")
        macro = np.load(tensor_dir / f"{split}_macro.npy")
        y_class = np.load(tensor_dir / f"{split}_y_class.npy")
        y_mag = np.load(tensor_dir / f"{split}_y_mag.npy")
        
        weights = None
        w_path = tensor_dir / f"{split}_weights.npy"
        if w_path.exists():
            weights = np.load(w_path)
            
        data[split] = {
            "micro": micro, "macro": macro, "y_class": y_class, "y_mag": y_mag, "weights": weights
        }
        print(f"  {split}: {len(y_class):,} samples")
    return data


def calibrate_and_save_config(fold_num, model_path, config_path):
    print(f"\n🔍 Threshold Calibration: Fold {fold_num}")
    tensor_dir = CV_DIR / f"fold_{fold_num}" / "tensors"
    
    micro = np.load(tensor_dir / "val_micro.npy")
    macro = np.load(tensor_dir / "val_macro.npy")
    y_class = np.load(tensor_dir / "val_y_class.npy")
    
    model = tf.keras.models.load_model(model_path)
    preds = model.predict([micro, macro], verbose=0)
    prob_win = preds[0].flatten()
    pred_os = preds[1].flatten()
    
    prob_th = 0.5
    prob_mask = (prob_win >= prob_th)
    
    best_wr = 0
    best_th = 1.0 # fallback
    
    os_thresholds = np.linspace(0.8, 2.0, 25)
    for th in os_thresholds:
        mask = prob_mask & (pred_os >= th)
        n_trades = np.sum(mask)
        # Demand at least 20 trades on the 6 month validation window
        if n_trades >= 20:
            wr = np.mean(y_class[mask])
            if wr > best_wr:
                best_wr = wr
                best_th = th
                
    print(f"🏆 Fold {fold_num} Best Pred_OS Threshold: {best_th:.2f} (WR: {best_wr:.2%})")

    config = {
        "Prob_Win_threshold": float(prob_th),
        "Pred_OS_threshold": float(round(best_th, 2)),
        "z_score_window": 1000,
        "micro_buffer_size": 100,
        "macro_history_size": 10
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"💾 Config saved to {config_path}")


def train_fold(fold_num):
    print("\n" + "=" * 60)
    print(f" TRAINING FOLD {fold_num}")
    print("=" * 60)
    
    fold_dir = CV_DIR / f"fold_{fold_num}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = fold_dir / "model.keras"
    config_path = fold_dir / "config.json"
    log_path = fold_dir / "training_log.csv"
    
    data = load_fold_tensors(fold_num)
    
    print("\n🏗️  Building fresh model...")
    model = build_model()
    model = compile_model(model)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE_ES, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=FACTOR_LR, patience=PATIENCE_LR, verbose=1, min_lr=1e-6),
        ModelCheckpoint(filepath=str(model_path), monitor='val_loss', save_best_only=True, verbose=1),
        CSVLogger(str(log_path)),
        OverfittingMonitor()
    ]
    
    print(f"\n🚀 Starting training Fold {fold_num}...")
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
    
    print(f"\n✅ Fold {fold_num} Training complete.")
    calibrate_and_save_config(fold_num, model_path, config_path)


def main():
    print("=" * 60)
    print(" IT2: 3-Fold Walk-Forward Cross Validation Training")
    print("=" * 60)
    
    for fold in [1, 2, 3]:
        train_fold(fold)
        
    print("\n🎉 ALL FOLDS TRAINED AND CALIBRATED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
