"""
2RR Overshot: Model Training
==============================
Trains the dual-head CNN+LSTM model on 2RR-labeled tensors.

Key difference from train_fallback.py:
  - Head A now predicts P(2-brick continuation) instead of P(1-brick win)
  - Class imbalance is more severe (~16% positive vs ~43% in 1RR)
  - Uses class_weight to compensate for the imbalanced labels

Reads from:  outputs/fallback_2rr/tensors/
Saves to:    outputs/fallback_2rr/model.keras
             outputs/fallback_2rr/training_log.csv
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
)

# Add src to path for model import
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from model import build_model, compile_model

# ── Configuration ──────────────────────────────────────────────
RR2_DIR = BASE_DIR / "outputs" / "fallback_2rr"
TENSOR_DIR = RR2_DIR / "tensors"
MODEL_PATH = RR2_DIR / "model.keras"
LOG_PATH = RR2_DIR / "training_log.csv"

BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE_ES = 15
PATIENCE_LR = 8
FACTOR_LR = 0.5


class OverfittingMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch < 20:
            return
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if val_loss and train_loss and val_loss > 1.5 * train_loss:
            print(f"\n⚠️  WARNING: Potential overfitting at epoch {epoch+1} "
                  f"(val_loss {val_loss:.4f} > 1.5 * train_loss {train_loss:.4f})")


def load_tensors():
    print("📂 Loading 2RR tensors...")

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

        n_pos = int(np.sum(y_class == 1.0))
        n_neg = int(np.sum(y_class == 0.0))
        print(f"  {split}: {len(y_class):,} samples "
              f"(2RR_WIN={n_pos:,} [{n_pos/len(y_class)*100:.1f}%], "
              f"2RR_LOSS={n_neg:,} [{n_neg/len(y_class)*100:.1f}%])")

    return data


def main():
    print("=" * 60)
    print(" 2RR Overshot: Model Training")
    print("=" * 60)

    # 1. Load Data
    data = load_tensors()

    # 2. Compute class weights to handle imbalance
    y_train = data["train"]["y_class"]
    n_pos = np.sum(y_train == 1.0)
    n_neg = np.sum(y_train == 0.0)
    n_total = len(y_train)

    # Balanced class weights: w_class = n_total / (2 * n_class)
    w_pos = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
    print(f"\n⚖️  Class weights: WIN={w_pos:.3f}, LOSS={w_neg:.3f}")

    # Apply class weights to sample weights
    sample_weights = data["train"]["weights"].copy() if data["train"]["weights"] is not None \
        else np.ones(n_total, dtype=np.float32)

    # Multiply existing sample weights by class weights
    sample_weights[y_train == 1.0] *= w_pos
    sample_weights[y_train == 0.0] *= w_neg

    # 3. Build & Compile Model
    print("\n🏗️  Building model...")
    model = build_model()
    model = compile_model(model)

    # 4. Setup Callbacks
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

    # 5. Train
    print(f"\n🚀 Starting 2RR training (max epochs={MAX_EPOCHS}, batch={BATCH_SIZE})...")
    history = model.fit(
        x=[data["train"]["micro"], data["train"]["macro"]],
        y=[data["train"]["y_class"], data["train"]["y_mag"]],
        validation_data=(
            [data["val"]["micro"], data["val"]["macro"]],
            [data["val"]["y_class"], data["val"]["y_mag"]]
        ),
        sample_weight=[sample_weights, sample_weights],
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print("\n✅ 2RR Training complete.")
    print(f"   Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"   Model saved to {MODEL_PATH}")
    print(f"   Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
