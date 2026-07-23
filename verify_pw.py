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

# Load test set
X_micro = np.load(TENSOR_DIR / "test_micro.npy")[:100]
X_macro = np.load(TENSOR_DIR / "test_macro.npy")[:100]

preds, _ = model.predict([X_micro, X_macro], verbose=0)
pw = preds.flatten()

print("Test Set Probabilities:")
print(f"Min: {pw.min():.4f}, Max: {pw.max():.4f}, Mean: {pw.mean():.4f}")

# Zero tensor
z_micro = np.zeros((100, 10, 100, 9), dtype=np.float32)
z_macro = np.zeros((100, 10, 7), dtype=np.float32)
preds_z, _ = model.predict([z_micro, z_macro], verbose=0)
pw_z = preds_z.flatten()

print("Zero Tensor Probabilities:")
print(f"Min: {pw_z.min():.4f}, Max: {pw_z.max():.4f}, Mean: {pw_z.mean():.4f}")
