"""
Phase 4.5: Signal Existence Checkpoint 2 (FR-SC-02)

Verifies that the engineered features carry measurable signal BEFORE
investing in the full CNN+LSTM training.

Algorithm:
  1. Load train and val tensors from Phase 4
  2. For each brick's micro tensor (10, 100, 9):
     - Take the most recent snapshot (index 9) → (100, 9)
     - Compute mean of each of 9 features across the LAST 10 ticks → 9D vector
  3. Train LogisticRegression(C=1.0) on 9D features → y_class
  4. Predict on validation set
  5. Decision gate based on accuracy:
     - RED  (<52%): Features carry almost no separable signal
     - GREEN (>55%): Strong signal, deep model should amplify
     - AMBER (52-55%): Weak signal, proceed but manage expectations

Output: outputs/signal_check_2.json
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path
import json
import sys

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"


# ═══════════════════════════════════════════════════════════════
# 4.5.1  Feature Aggregation
# ═══════════════════════════════════════════════════════════════

def aggregate_features(micro: np.ndarray) -> np.ndarray:
    """Aggregate micro tensors into 9D feature vectors.

    For each sample:
      - Take the most recent snapshot (index 9) → (100, 9)
      - Compute mean of each feature across the LAST 10 ticks → (9,)

    Args:
        micro: (N, 10, 100, 9) micro tensor

    Returns:
        (N, 9) feature matrix
    """
    # Most recent snapshot is at index 9 (last of 10 bricks)
    latest_snapshot = micro[:, -1, :, :]  # (N, 100, 9)

    # Last 10 ticks (rows 90-99)
    last_10 = latest_snapshot[:, -10:, :]  # (N, 10, 9)

    # Mean across tick dimension
    features = np.mean(last_10, axis=1)  # (N, 9)

    return features


# ═══════════════════════════════════════════════════════════════
# 4.5.2  Logistic Regression Baseline
# ═══════════════════════════════════════════════════════════════

def run_logistic_baseline(X_train, y_train, X_val, y_val, train_weights=None):
    """Train LogisticRegression and evaluate on validation set.

    Returns:
        dict with accuracy, AUC, majority baseline, feature importances
    """
    # Train
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train, sample_weight=train_weights)

    # Predict
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    # Majority-class baseline
    majority_class = 1.0 if np.mean(y_train) >= 0.5 else 0.0
    majority_accuracy = accuracy_score(y_val, np.full_like(y_val, majority_class))

    # Feature importances (coefficient magnitudes)
    feature_names = ["z_OFI", "z_Depth", "z_Susc", "z_Vel", "z_Spread",
                     "Progress", "Flag_Curr", "Flag_Zone", "Decay"]
    importances = {name: round(float(abs(c)), 6)
                   for name, c in zip(feature_names, clf.coef_[0])}

    return {
        "accuracy": round(float(accuracy), 4),
        "auc": round(float(auc), 4),
        "majority_baseline": round(float(majority_accuracy), 4),
        "lift_over_majority": round(float(accuracy - majority_accuracy), 4),
        "feature_importances": importances,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }


# ═══════════════════════════════════════════════════════════════
# 4.5.3  Decision Gate
# ═══════════════════════════════════════════════════════════════

def apply_decision_gate(accuracy: float) -> str:
    """Apply accuracy-based decision gate."""
    if accuracy > 0.55:
        return "GREEN"
    elif accuracy >= 0.52:
        return "AMBER"
    else:
        return "RED"


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Phase 4.5: Signal Existence Checkpoint 2")
    print("=" * 60)

    # Check prerequisites
    for f in ["train_micro.npy", "train_y_class.npy", "val_micro.npy", "val_y_class.npy"]:
        if not (TENSOR_DIR / f).exists():
            print(f"ERROR: {f} not found. Run Phase 4 first.")
            sys.exit(1)

    # Load tensors
    print("\n📂 Loading tensors...")
    train_micro = np.load(TENSOR_DIR / "train_micro.npy")
    train_y = np.load(TENSOR_DIR / "train_y_class.npy")
    val_micro = np.load(TENSOR_DIR / "val_micro.npy")
    val_y = np.load(TENSOR_DIR / "val_y_class.npy")

    train_weights = None
    weights_path = TENSOR_DIR / "train_weights.npy"
    if weights_path.exists():
        train_weights = np.load(weights_path)

    print(f"  Train: {train_micro.shape[0]:,} samples")
    print(f"  Val:   {val_micro.shape[0]:,} samples")

    # Aggregate features
    print("\n📊 Aggregating features (mean of last 10 ticks in most recent snapshot)...")
    X_train = aggregate_features(train_micro)
    X_val = aggregate_features(val_micro)
    print(f"  Feature matrix: train={X_train.shape}, val={X_val.shape}")

    # Feature statistics
    feature_names = ["z_OFI", "z_Depth", "z_Susc", "z_Vel", "z_Spread",
                     "Progress", "Flag_Curr", "Flag_Zone", "Decay"]
    print("\n📊 Feature statistics (train):")
    print(f"  {'Feature':12s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    for j, name in enumerate(feature_names):
        col = X_train[:, j]
        print(f"  {name:12s}  {col.mean():8.4f}  {col.std():8.4f}  {col.min():8.4f}  {col.max():8.4f}")

    # Logistic regression baseline
    print("\n🔬 Training LogisticRegression(C=1.0)...")
    results = run_logistic_baseline(X_train, train_y, X_val, val_y, train_weights)

    print(f"\n  Accuracy:          {results['accuracy']:.4f}")
    print(f"  AUC:               {results['auc']:.4f}")
    print(f"  Majority baseline: {results['majority_baseline']:.4f}")
    print(f"  Lift over majority: {results['lift_over_majority']:+.4f}")

    print("\n  Feature importances (|coef|):")
    sorted_feats = sorted(results["feature_importances"].items(), key=lambda x: -x[1])
    for name, imp in sorted_feats:
        bar = "█" * int(imp * 50)
        print(f"    {name:12s}  {imp:.6f}  {bar}")

    # Decision gate
    decision = apply_decision_gate(results["accuracy"])

    print("\n" + "=" * 60)
    if decision == "GREEN":
        print("🟢 GREEN — Strong signal detected (accuracy > 55%)")
        print("   Deep model should amplify this. Proceed with confidence.")
    elif decision == "AMBER":
        print("🟡 AMBER — Weak signal detected (accuracy 52-55%)")
        print("   CNN+LSTM may find non-linear patterns. Proceed with")
        print("   managed expectations.")
    else:
        print("🔴 RED — Features carry almost no separable signal (<52%)")
        print("   Consider reviewing feature engineering before training")
        print("   the deep model.")
    print("=" * 60)

    # Save results
    output = {
        "checkpoint": "signal_check_2",
        "results": results,
        "decision": decision,
        "thresholds": {
            "green": "accuracy > 55%",
            "amber": "52% <= accuracy <= 55%",
            "red": "accuracy < 52%",
        },
    }

    out_path = OUTPUT_DIR / "signal_check_2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved to {out_path}")

    return output


if __name__ == "__main__":
    main()
