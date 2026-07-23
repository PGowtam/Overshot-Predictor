import sys
import os
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent))

from label_generator import generate_all_labels
from feature_engine import process_all_ticks

def main():
    # 1. Generate Labels for new Renko holdout
    renko_path = "Data/Processed/XAUUSD_Holdout_K00295.csv"
    print(f"--- Generating Labels for {renko_path} ---")
    
    # We use pricing_mode='execution' to be as realistic as possible
    # We'll save it to a specific holdout parquet
    df_labels = generate_all_labels(renko_csv_path=renko_path, pricing_mode='execution')
    
    # Save to a temporary location that feature_engine will pick up
    # (Existing feature_engine hardcodes OUTPUT_DIR / "labels.parquet")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup existing labels if they exist
    main_labels_path = output_dir / "labels.parquet"
    backup_path = output_dir / "labels_backup.parquet"
    if main_labels_path.exists():
        os.rename(main_labels_path, backup_path)
        print("Backed up existing labels.parquet")
    
    # Save new labels (cast sequence to string to avoid OverflowError in pyarrow)
    if "sequence" in df_labels.columns:
        df_labels["sequence"] = df_labels["sequence"].astype(str)
    df_labels.to_parquet(main_labels_path, index=False)
    print(f"Saved temporary labels.parquet for feature extraction")
    
    # 2. Generate Features
    print("\n--- Generating Features ---")
    feat_stats, nan_inf_count = process_all_ticks()
    
    # 3. Clean up and Restore
    print("\n--- Finalizing Holdout Data ---")
    
    # Move features to a dedicated holdout directory
    feature_dir = output_dir / "features"
    holdout_feature_dir = output_dir / "features_holdout_K295"
    if holdout_feature_dir.exists():
        import shutil
        shutil.rmtree(holdout_feature_dir)
    os.rename(feature_dir, holdout_feature_dir)
    
    # Move holdout labels to dedicated path
    holdout_labels_path = output_dir / "labels_holdout_K295.parquet"
    os.rename(main_labels_path, holdout_labels_path)
    
    # Restore backup
    if backup_path.exists():
        os.rename(backup_path, main_labels_path)
        print("Restored original labels.parquet")
    
    print(f"\n✅ Holdout features generated in: {holdout_feature_dir}")
    print(f"✅ Holdout labels saved to: {holdout_labels_path}")

if __name__ == "__main__":
    main()
