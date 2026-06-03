import sys, os
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path("/Users/gopo/Quant Projects/CAPSTONE/Overshot")
sys.path.insert(0, str(BASE_DIR / "src"))
from models_exec import build_baseline_exec_model

TENSOR_DIR = BASE_DIR / "outputs" / "exec_tensors"
MODEL_DIR = BASE_DIR / "outputs" / "exec_baseline"

model = build_baseline_exec_model()
model.load_weights(str(MODEL_DIR / "model.keras"))

X_micro = np.load(TENSOR_DIR / "test_micro.npy")
X_macro = np.load(TENSOR_DIR / "test_macro.npy")
y_class = np.load(TENSOR_DIR / "test_y_class.npy").flatten()

preds, _ = model.predict([X_micro, X_macro], batch_size=2048, verbose=0)
prob_win = preds.flatten()
y_fade = 1.0 - y_class

print("\n=== GLOBAL FADE WIN RATES (BASELINE MODEL) ===")
for thresh in [0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25]:
    mask = prob_win <= thresh
    n = mask.sum()
    if n > 0:
        wr = y_fade[mask].mean()
        print(f"Fade if prob <= {thresh:.2f}: n={n:4d}, WinRate={wr:.2%}")
