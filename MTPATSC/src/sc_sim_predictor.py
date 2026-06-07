"""
sc_sim_predictor.py
Isolated subprocess: loads the keras inference model, runs predictions on the
2026 bricks parquet, and saves probabilities to a .npy file.
Running TF in complete isolation avoids the macOS Metal/OpenMP deadlock.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.set_visible_devices([], 'MPS')
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Predictor")

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "setup_classifier"

def main():
    if len(sys.argv) > 2:
        bricks_path = Path(sys.argv[1])
        probs_path = Path(sys.argv[2])
    else:
        bricks_path = OUTPUT_DIR / "sim_2026_bricks.parquet"
        probs_path  = OUTPUT_DIR / "sim_2026_probs.npy"
        
    model_path  = OUTPUT_DIR / "model_inference.keras"
    scaler_path = OUTPUT_DIR / "scalar_scaler.pkl"

    if not bricks_path.exists():
        logger.error(f"Bricks parquet not found: {bricks_path}")
        sys.exit(1)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    if not scaler_path.exists():
        logger.error(f"Scaler not found: {scaler_path}")
        sys.exit(1)

    logger.info(f"Loading bricks from {bricks_path}...")
    df = pd.read_parquet(bricks_path)
    logger.info(f"  {len(df)} bricks loaded.")

    logger.info("Assembling tensors...")
    ancs_fine  = np.stack([np.stack(r)  for r in df["ancs_fine"]]).astype(np.float32)
    ancs_coarse= np.stack([np.stack(r)  for r in df["ancs_coarse"]]).astype(np.float32)
    history    = np.stack([np.stack([np.stack(x) for x in r]) for r in df["history"]]).astype(np.float32)
    candle     = np.stack([np.array(r, dtype=np.float32) for r in df["candle_features"]])
    momentum   = np.stack([np.array(r, dtype=np.float32) for r in df["momentum"]])
    scalars    = np.hstack([candle, momentum])

    logger.info("Scaling scalars...")
    scaler = joblib.load(scaler_path)
    scaled_scalars = scaler.transform(scalars).astype(np.float32)

    logger.info("Loading inference model...")
    model = tf.keras.models.load_model(str(model_path))

    logger.info("Running predictions (this may take a moment)...")
    try:
        # For small datasets, model.predict can deadlock on macOS Metal
        probs = model([ancs_fine, ancs_coarse, history, scaled_scalars], training=False).numpy()
    except Exception as e:
        logger.warning(f"Direct inference failed ({e}), falling back to predict...")
        probs = model.predict([ancs_fine, ancs_coarse, history, scaled_scalars], verbose=1)

    np.save(str(probs_path), probs)
    logger.info(f"Saved probabilities ({probs.shape}) to {probs_path}")

if __name__ == "__main__":
    main()
