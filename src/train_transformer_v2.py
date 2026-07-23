"""
Train Transformer V2
====================
Trains the dual-branch Selective Trade Discovery System network on the 
dynamically extracted V2 tensors.
"""

import os
import logging
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from models_exec_v2 import build_transformer_exec_model, compile_transformer_model

BASE_DIR = Path(__file__).resolve().parent.parent
TENSOR_DIR = BASE_DIR / "outputs" / "exec_tensors_v2"
OUTPUT_DIR = BASE_DIR / "outputs" / "transformer_v2"

logger = logging.getLogger(__name__)

def load_data(split_name):
    logger.info(f"Loading {split_name} data...")
    micro = np.load(TENSOR_DIR / f"{split_name}_micro.npy")     # (N, 100, 9)
    macro = np.load(TENSOR_DIR / f"{split_name}_macro.npy")     # (N, 10, 11)
    summary = np.load(TENSOR_DIR / f"{split_name}_summary.npy") # (N, 5)
    y_class = np.load(TENSOR_DIR / f"{split_name}_y_class.npy") # (N, 1)
    y_mag = np.load(TENSOR_DIR / f"{split_name}_y_mag.npy")     # (N, 1)
    
    y_class = y_class.flatten()
    y_mag = y_mag.flatten()
    
    # Filter NaNs
    valid_mask = ~np.isnan(y_class) & ~np.isnan(y_mag)
    
    X = {
        'micro_input': micro[valid_mask],
        'macro_input': macro[valid_mask],
        'summary_input': summary[valid_mask]
    }
    
    y = {
        'prob_win': y_class[valid_mask],
        'pred_os': y_mag[valid_mask]
    }
    
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(OUTPUT_DIR / "train.log"), mode='w')
        ]
    )
    
    # 1. Load Data
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")
    
    logger.info(f"Train samples: {len(y_train['prob_win'])}")
    logger.info(f"Val samples: {len(y_val['prob_win'])}")
    
    # 2. Build Model
    model = build_transformer_exec_model()
    model = compile_transformer_model(model)
    model.summary(print_fn=logger.info)
    
    # 3. Callbacks
    # Using ReduceLROnPlateau for safety over custom schedules to prevent hangs
    callbacks = [
        ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "model_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 4. Train
    logger.info("Starting Transformer Training...")
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True, # Shuffle at the brick level within the training window
        verbose=1
    )
    
    # 5. Final Evaluation
    logger.info("Evaluating on Val Set...")
    eval_results = model.evaluate(X_val, y_val, batch_size=args.batch_size)
    for name, val in zip(model.metrics_names, eval_results):
        logger.info(f"{name}: {val:.4f}")
        
    model.save(str(OUTPUT_DIR / "model_final.keras"))
    logger.info("Training Complete. Model saved.")

if __name__ == "__main__":
    main()
