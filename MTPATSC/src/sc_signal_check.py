import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
TENSOR_DIR = BASE_DIR / "outputs" / "setup_classifier" / "tensors"

def main():
    train_path = TENSOR_DIR / "train.npz"
    val_path = TENSOR_DIR / "val.npz"
    
    if not train_path.exists() or not val_path.exists():
        print(f"❌ Tensor files not found in {TENSOR_DIR}. Run sc_tensor_builder.py first.")
        return
        
    print("Loading tensors for baseline signal check...")
    train = np.load(train_path)
    val = np.load(val_path)
    
    X_train, y_train = train['scalars'], train['label']
    X_val, y_val = val['scalars'], val['label']
    
    print(f"Train scalars shape: {X_train.shape}")
    print(f"Val scalars shape: {X_val.shape}")
    
    print("\nTraining Logistic Regression baseline on scalar features...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_val)
    
    print("\n--- Validation Classification Report (Baseline) ---")
    report = classification_report(y_val, y_pred, target_names=["T0", "T1", "T2", "T3", "T4"])
    print(report)
    
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    if macro_f1 > 0.21:
        print("✅ GO: Signal check passed! Macro F1 is above the 0.21 baseline threshold.")
    elif macro_f1 < 0.20:
        print("❌ ABORT: Signal check failed! Macro F1 is below the 0.20 random guess threshold. Investigate features.")
    else:
        print("⚠️ WARNING: Signal check marginal (between 0.20 and 0.21). Model may have difficulty learning.")

if __name__ == "__main__":
    main()
