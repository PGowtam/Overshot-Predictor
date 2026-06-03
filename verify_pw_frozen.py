import sys, os
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path("/Users/gopo/Quant Projects/CAPSTONE/Overshot")
sys.path.insert(0, str(BASE_DIR / "src"))
from models_exec import build_attention_exec_model, HIGH_MI_DIMS

TENSOR_DIR = BASE_DIR / "outputs" / "exec_tensors_regime"
MODEL_DIR = BASE_DIR / "outputs" / "exec_attn_regime"

model = build_attention_exec_model(high_mi_dims=HIGH_MI_DIMS)
model.load_weights(str(MODEL_DIR / "model.keras"))

X_micro = np.load(TENSOR_DIR / "test_micro.npy")
X_macro = np.load(TENSOR_DIR / "test_macro.npy")

# FREEZE LOCAL WR
X_macro[:, :, 6] = 0.3533

preds, _ = model.predict([X_micro, X_macro], batch_size=2048, verbose=0)
pw = preds.flatten()

print("Test Set Probabilities (FROZEN local_wr):")
print(f"Min: {pw.min():.4f}, Max: {pw.max():.4f}, Mean: {pw.mean():.4f}")
