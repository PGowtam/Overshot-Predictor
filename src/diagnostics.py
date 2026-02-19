"""
Phase 6: Post-training diagnostics (FR-TR-07 verification)

Performs detailed model diagnostics:
1. Head A/B prediction variance/std on validation set.
2. Check for Head B collapse (std < 0.05).
3. Mean Pred_OS on training LOSS samples (sanity check for regression head).
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add src to path just in case
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"
MODEL_PATH = OUTPUT_DIR / "model.keras"

def load_data(split):
    print(f"📂 Loading {split} tensors...")
    micro = np.load(TENSOR_DIR / f"{split}_micro.npy")
    macro = np.load(TENSOR_DIR / f"{split}_macro.npy")
    y_class = np.load(TENSOR_DIR / f"{split}_y_class.npy")
    y_mag = np.load(TENSOR_DIR / f"{split}_y_mag.npy")
    return micro, macro, y_class, y_mag

def run_diagnostics():
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    # 1. Load Model
    print("MATCHING MODEL...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Validation Diagnostics
    micro_val, macro_val, _, _ = load_data("val")
    
    print("\n🔮 Running predictions on Validation set...")
    # List-based input to match training format
    preds = model.predict([micro_val, macro_val], verbose=1)
    
    # Unpack predictions (Head A, Head B)
    prob_win_val = preds[0].flatten()
    pred_os_val = preds[1].flatten()
    
    # Head A Variance
    var_a = np.var(prob_win_val)
    std_a = np.std(prob_win_val)
    print(f"\n📊 Head A (Prob_Win) Validation Stats:")
    print(f"   Variance: {var_a:.6f}")
    print(f"   Std Dev:  {std_a:.6f}")
    print(f"   Mean:     {np.mean(prob_win_val):.4f}")
    print(f"   Range:    [{np.min(prob_win_val):.4f}, {np.max(prob_win_val):.4f}]")
    
    # Head B Variance
    var_b = np.var(pred_os_val)
    std_b = np.std(pred_os_val)
    print(f"\n📊 Head B (Pred_OS) Validation Stats:")
    print(f"   Variance: {var_b:.6f}")
    print(f"   Std Dev:  {std_b:.6f}")
    print(f"   Mean:     {np.mean(pred_os_val):.4f}")
    print(f"   Range:    [{np.min(pred_os_val):.4f}, {np.max(pred_os_val):.4f}]")
    
    if std_b < 0.05:
        print("⚠️  WARNING: Head B std < 0.05. Regression output may have collapsed!")
    else:
        print("✅ Head B std >= 0.05. No obvious collapse.")

    # 3. Training LOSS Samples Check
    # We only need y_class and inputs for training to filter LOSS samples
    # To save memory, we load training data now, after val is done (though Python GC might not be instant)
    micro_train, macro_train, y_class_train, _ = load_data("train")
    
    # Identify LOSS samples (y_class == 0)
    loss_indices = np.where(y_class_train == 0)[0]
    print(f"\n📉 Analyzing {len(loss_indices):,} LOSS samples from Training set...")
    
    if len(loss_indices) > 0:
        # Filter inputs
        micro_loss = micro_train[loss_indices]
        macro_loss = macro_train[loss_indices]
        
        # Predict
        preds_loss = model.predict([micro_loss, macro_loss], verbose=1)
        pred_os_loss = preds_loss[1].flatten()
        
        mean_os_loss = np.mean(pred_os_loss)
        print(f"   Mean Pred_OS on LOSS samples: {mean_os_loss:.4f}")
        
        if mean_os_loss < 0.5:
             print("✅ Mean Pred_OS < 0.5. Model correctly identifies LOSS samples as low magnitude.")
        else:
             print("⚠️  Mean Pred_OS >= 0.5. Model predicts high magnitude even for LOSS samples.")
    else:
        print("   No LOSS samples found in training set?")

if __name__ == "__main__":
    run_diagnostics()
