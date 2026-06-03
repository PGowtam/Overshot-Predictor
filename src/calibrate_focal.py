import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

MODEL_DIR = BASE_DIR / "outputs" / "exec_focal"
TENSOR_DIR = BASE_DIR / "outputs" / "exec_tensors"

def load_split(split):
    micro = np.load(TENSOR_DIR / f"{split}_micro.npy")
    macro = np.load(TENSOR_DIR / f"{split}_macro.npy")
    y_class = np.load(TENSOR_DIR / f"{split}_y_class.npy").flatten()
    return micro, macro, y_class

def main():
    print("Loading Focal Model...")
    model_path = MODEL_DIR / "model.keras"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}. Train first.")
        return
        
    model = tf.keras.models.load_model(str(model_path), compile=False)
    
    print("Loading Tensors...")
    micro_val, macro_val, y_val = load_split("val")
    micro_test, macro_test, y_test = load_split("test")
    
    print("Predicting Validation Set...")
    prob_val, _ = model.predict([micro_val, macro_val], batch_size=512, verbose=0)
    prob_val = prob_val.flatten()
    
    print("Predicting Test Set...")
    prob_test, _ = model.predict([micro_test, macro_test], batch_size=512, verbose=0)
    prob_test = prob_test.flatten()
    
    print("\n--- Diagnostic: Check if Calibration is Warranted ---")
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_val, prob_val, n_bins=10, strategy='quantile'
    )
    
    print("Calibration Curve (Validation Set):")
    for f, m in zip(fraction_of_positives, mean_predicted_value):
        print(f"  Predicted Bin Mean: {m:.4f} -> True Pos Fraction: {f:.4f}")
        
    # Check if the line is completely flat
    if np.max(fraction_of_positives) - np.min(fraction_of_positives) < 0.05:
        print("\n[WARNING] The calibration curve is flat. The model outputs have NO real discriminative spread.")
        print("          Platt Scaling will just crush the probabilities. Exiting.")
        return
        
    print("\n--- Calibration (Platt Scaling) ---")
    calibrator = LogisticRegression()
    # Reshape for sklearn
    X_val = prob_val.reshape(-1, 1)
    X_test = prob_test.reshape(-1, 1)
    
    # Fit calibrator on Val
    calibrator.fit(X_val, y_val)
    
    # Save calibrator
    joblib.dump(calibrator, MODEL_DIR / "platt_scaler.pkl")
    print("Saved Platt Scaler to outputs/exec_focal/platt_scaler.pkl")
    
    # Predict calibrated probs
    cal_prob_test = calibrator.predict_proba(X_test)[:, 1]
    
    print("\n--- Test Set Probability Distributions ---")
    print(f"Raw Prob_Win: Mean={np.mean(prob_test):.4f}, Max={np.max(prob_test):.4f}, Min={np.min(prob_test):.4f}")
    print(f"Cal Prob_Win: Mean={np.mean(cal_prob_test):.4f}, Max={np.max(cal_prob_test):.4f}, Min={np.min(cal_prob_test):.4f}")
    
    print("\n--- Threshold Feasibility (Calibrated) ---")
    # Let's check some high thresholds on the calibrated output
    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = (cal_prob_test >= thresh)
        count = np.sum(mask)
        if count > 0:
            wins = np.sum(y_test[mask])
            wr = wins / count
            print(f"Threshold >= {thresh:.2f} | Trades: {count:4d} | Win Rate: {wr:.2%}")
        else:
            print(f"Threshold >= {thresh:.2f} | Trades:    0 | Win Rate: N/A")
            
    print("\nCalibration Complete!")

if __name__ == "__main__":
    main()
