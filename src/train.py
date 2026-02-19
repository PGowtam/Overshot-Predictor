"""
Phase 6: Training (FR-TR-01 to FR-TR-07)

Trains the dual-head CNN+LSTM model using Phase 4 tensors.

Features:
  - Hybrid loss: BCE (Head A) + 0.3 * Huber (Head B)
  - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
  - Overfitting monitor: Checks val_loss > 1.5 * train_loss after epoch 20
  - Data: Loads pre-built tensors from outputs/tensors/

Usage:
  python src/train.py
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
)
from tensorflow.keras.losses import BinaryCrossentropy, Huber

# Add src to path for model import
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from model import build_model, compile_model

# ── Configuration ──────────────────────────────────────────────
OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"
MODEL_PATH = OUTPUT_DIR / "model.keras"
LOG_PATH = OUTPUT_DIR / "training_log.csv"

BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE_ES = 15
PATIENCE_LR = 8
FACTOR_LR = 0.5


# ═══════════════════════════════════════════════════════════════
# Custom Callback: Overfitting Monitor
# ═══════════════════════════════════════════════════════════════

class OverfittingMonitor(Callback):
    """Flags potential overfitting if val_loss spikes relative to train_loss."""
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch < 20:
            return
            
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        
        if val_loss and train_loss and val_loss > 1.5 * train_loss:
            print(f"\n⚠️  WARNING: Potential overfitting detected at epoch {epoch+1} "
                  f"(val_loss {val_loss:.4f} > 1.5 * train_loss {train_loss:.4f})")


# ═══════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════

def load_tensors():
    """Load train/val tensors."""
    print("📂 Loading tensors...")
    
    data = {}
    for split in ["train", "val"]:
        micro = np.load(TENSOR_DIR / f"{split}_micro.npy")
        macro = np.load(TENSOR_DIR / f"{split}_macro.npy")
        y_class = np.load(TENSOR_DIR / f"{split}_y_class.npy")
        y_mag = np.load(TENSOR_DIR / f"{split}_y_mag.npy")
        
        weights = None
        if split == "train":
            w_path = TENSOR_DIR / "train_weights.npy"
            if w_path.exists():
                weights = np.load(w_path)
        
        data[split] = {
            "micro": micro,
            "macro": macro,
            "y_class": y_class,
            "y_mag": y_mag,
            "weights": weights
        }
        print(f"  {split}: {len(y_class):,} samples")
        
    return data


def main():
    print("=" * 60)
    print(" Phase 6: Model Training")
    print("=" * 60)
    
    # 1. Load Data
    data = load_tensors()
    
    # 2. Build & Compile Model
    print("\n🏗️  Building model...")
    model = build_model()
    model = compile_model(model)
    
    # 3. Setup Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=PATIENCE_ES, 
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=FACTOR_LR, 
            patience=PATIENCE_LR, 
            verbose=1,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(str(LOG_PATH)),
        OverfittingMonitor()
    ]
    
    # 4. Train
    print(f"\n🚀 Starting training (max epochs={MAX_EPOCHS}, batch={BATCH_SIZE})...")
    history = model.fit(
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
    print(f"   Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"   Model saved to {MODEL_PATH}")
    print(f"   Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
