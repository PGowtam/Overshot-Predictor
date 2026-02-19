"""
Phase 6: Post-training diagnostics (FR-TR-07 verification)

Reads outputs/training_log.csv and plots loss curves.
Saves plot to outputs/plots/loss_curves.png.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "outputs" / "training_log.csv"
PLOT_DIR = BASE_DIR / "outputs" / "plots"

def plot_loss_curves():
    if not LOG_PATH.exists():
        print(f"❌ Log file not found at {LOG_PATH}")
        return

    df = pd.read_csv(LOG_PATH)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Total Loss
    plt.plot(df['epoch'], df['loss'], label='Train Loss', color='blue')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='orange')
    
    # Check for overfitting threshold (1.5x)
    # We can plot a dashed line for 1.5 * Train Loss? No, too messy.
    
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Hybrid)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "loss_curves.png"
    plt.savefig(out_path)
    print(f"✅ Loss curve saved to {out_path}")
    
    # Print metrics at best epoch
    best_epoch_idx = df['val_loss'].idxmin()
    best_row = df.iloc[best_epoch_idx]
    
    print("\n📊 Best Epoch Metrics:")
    print(f"  Epoch: {best_row['epoch']}")
    print(f"  Val Loss: {best_row['val_loss']:.4f}")
    print(f"  Val Acc (Head A): {best_row['val_prob_win_accuracy']:.2%}")
    print(f"  Val MAE (Head B): {best_row['val_pred_os_mae']:.4f}")

if __name__ == "__main__":
    plot_loss_curves()
