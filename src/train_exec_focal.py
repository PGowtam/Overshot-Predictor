"""
Train Execution Baseline Model
==============================
Trains Model A: The standard dual-head architecture using execution labels.
Outputs to: outputs/exec_baseline/

Uses the same training pattern as train.py (raw numpy arrays, no DataGenerator).
"""

import sys
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
)

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from tensorflow.keras.losses import BinaryFocalCrossentropy, Huber
from tensorflow.keras.optimizers import Adam
from models_exec import build_baseline_exec_model

OUTPUT_DIR = BASE_DIR / "outputs" / "exec_focal"
TENSOR_DIR = BASE_DIR / "outputs" / "exec_tensors"

MODEL_PATH = OUTPUT_DIR / "model.keras"
LOG_PATH = OUTPUT_DIR / "training_log.csv"
REPORT_PATH = OUTPUT_DIR / "report.md"

BATCH_SIZE = 64
MAX_EPOCHS = 200


class OverfittingMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch < 20: return
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if val_loss and train_loss and val_loss > 1.5 * train_loss:
            print(f"\n⚠️  WARNING: Potential overfitting detected at epoch {epoch+1}")


def generate_markdown_report(history):
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']
    best_epoch = np.argmin(val_loss) + 1
    
    report = f"""# Focal Loss Execution Model - Training Report

## Configuration
- **Model**: Dual-Head CNN+LSTM
- **Loss**: BinaryFocalCrossentropy (gamma=2.0, alpha=0.75)
- **Data Source**: Imbalanced Prior
- **Batch Size**: {BATCH_SIZE}
- **Epochs Ran**: {len(val_loss)}
- **Best Epoch**: {best_epoch}

## Results
- **Best Validation Loss**: `{np.min(val_loss):.4f}`
- **Final Train Loss**: `{train_loss[-1]:.4f}`

*(Model saved to outputs/exec_focal/model.keras)*
"""
    with open(REPORT_PATH, 'w') as f:
        f.write(report)


def load_tensors():
    print("📂 Loading execution tensors...")
    data = {}
    for split in ["train", "val"]:
        micro = np.load(TENSOR_DIR / f"{split}_micro.npy")
        macro = np.load(TENSOR_DIR / f"{split}_macro.npy")
        y_class = np.load(TENSOR_DIR / f"{split}_y_class.npy")
        y_mag = np.load(TENSOR_DIR / f"{split}_y_mag.npy")
        
        data[split] = {
            "micro": micro, "macro": macro,
            "y_class": y_class, "y_mag": y_mag
        }
        print(f"  {split}: {len(y_class):,} samples")
    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    data = load_tensors()
    
    print("\n🏗️  Building Focal Loss Execution Model...")
    model = build_baseline_exec_model()
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={
            'prob_win': BinaryFocalCrossentropy(gamma=2.0),
            'pred_os': Huber(delta=1.0)
        },
        loss_weights={
            'prob_win': 1.0,
            'pred_os': 0.3
        },
        metrics={
            'prob_win': 'accuracy',
            'pred_os': 'mae'
        }
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(MODEL_PATH), monitor='val_loss', save_best_only=True, verbose=1),
        CSVLogger(str(LOG_PATH)),
        OverfittingMonitor()
    ]
    
    print("\n🚀 Training model...")
    # Keras does not support class_weight for multi-output models, must use sample_weight
    w_train = np.where(data["train"]["y_class"] == 1, 0.6467 / 0.3533, 1.0)
    
    w_val = np.ones_like(data["val"]["y_class"])
    
    history = model.fit(
        x=[data["train"]["micro"], data["train"]["macro"]],
        y=[data["train"]["y_class"], data["train"]["y_mag"]],
        validation_data=(
            [data["val"]["micro"], data["val"]["macro"]],
            [data["val"]["y_class"], data["val"]["y_mag"]],
            [w_val, w_val]
        ),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        sample_weight=[w_train, np.ones_like(w_train)],
        verbose=1
    )
    
    generate_markdown_report(history)
    print(f"\n✅ Training complete! Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
