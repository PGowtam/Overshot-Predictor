"""
Phase 8.1b: Process Holdout Data (Pipeline)

One-shot pipeline to create tensors for the 2024 Holdout dataset.
1. Labels: Generates labels from 'renko_with_tick_outcomes_no_be_24_local.csv'.
2. Features: Computes 9D + 3D features using local ticks.
3. Buffers: Simulates micro-buffers.
4. Tensors: Builds classification tensors (forced to 'holdout' split).
5. Deploy: Moves tensors to 'outputs/tensors' for evaluation.
"""

import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import importlib

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Paths
DATA_DIR = BASE_DIR / "Data" / "Raw"
CSV_PATH = BASE_DIR / "Data" / "Processed" / "XAUUSD_Holdout_K00295.csv"
TICK_DIR = DATA_DIR / "Ticks"

HOLDOUT_DIR = BASE_DIR / "outputs" / "holdout_K295"
HOLDOUT_FEATURES = HOLDOUT_DIR / "features"
HOLDOUT_TENSORS = HOLDOUT_DIR / "tensors"
MAIN_TENSOR_DIR = BASE_DIR / "outputs" / "tensors_holdout_K295"

def step_1_labels():
    print("\n" + "="*50)
    print(" 1. GENERATING LABELS (Holdout)")
    print("="*50)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    
    import label_generator as lg
    
    # Generate
    # Note: lg.main() writes to lg.OUTPUT_DIR / labels.parquet
    # We will invoke generate_all_labels directly to get DF, then save manually
    df = lg.generate_all_labels(renko_csv_path=str(CSV_PATH), tick_dir=str(TICK_DIR))
    
    # Save to holdout dir
    out_path = HOLDOUT_DIR / "labels.parquet"
    if "sequence" in df.columns:
        df["sequence"] = df["sequence"].astype(str)
    df.to_parquet(out_path, index=False)
    print(f"💾 Saved {len(df)} labels to {out_path}")

def step_2_features():
    print("\n" + "="*50)
    print(" 2. FEATURE ENGINEERING (Holdout)")
    print("="*50)
    HOLDOUT_FEATURES.mkdir(parents=True, exist_ok=True)
    
    import feature_engine as fe
    
    # Monkey-patch paths
    fe.OUTPUT_DIR = HOLDOUT_DIR
    fe.FEATURE_DIR = HOLDOUT_FEATURES
    fe.TICK_DIR = TICK_DIR
    
    # Run
    fe.main()

def step_3_buffers():
    print("\n" + "="*50)
    print(" 3. BUFFER SIMULATION (Holdout)")
    print("="*50)
    
    import buffer_sim as bs
    
    # Run (accepts arg)
    bs.simulate_buffers(feature_dir=HOLDOUT_FEATURES)
    bs.validate_buffers(feature_dir=HOLDOUT_FEATURES)

def step_4_tensors():
    print("\n" + "="*50)
    print(" 4. TENSOR CONSTRUCTION (Holdout)")
    print("="*50)
    HOLDOUT_TENSORS.mkdir(parents=True, exist_ok=True)
    
    import tensor_builder as tb
    
    # Monkey-patch paths
    tb.OUTPUT_DIR = HOLDOUT_DIR
    tb.FEATURE_DIR = HOLDOUT_FEATURES
    tb.SNAPSHOT_DIR = HOLDOUT_FEATURES / "snapshots"
    tb.TENSOR_DIR = HOLDOUT_TENSORS
    
    # Monkey-patch split assignment to force 'holdout'
    tb.assign_split = lambda date: "holdout"
    
    # Run
    tb.build_and_save_tensors()

def step_5_deploy():
    print("\n" + "="*50)
    print(" 5. DEPLOYING TENSORS")
    print("="*50)
    
    # Copy holdout_*.npy from HOLDOUT_TENSORS to MAIN_TENSOR_DIR
    for f in HOLDOUT_TENSORS.glob("holdout_*.npy"):
        target = MAIN_TENSOR_DIR / f.name
        shutil.copy2(f, target)
        print(f"✅ Deployed: {target}")

def main():
    if not CSV_PATH.exists():
        print(f"❌ Input CSV not found: {CSV_PATH}")
        return
        
    step_1_labels()
    step_2_features()
    step_3_buffers()
    step_4_tensors()
    step_5_deploy()
    
    print("\n🚀 Holdout pipeline complete. Now run 'src/evaluate.py'.")

if __name__ == "__main__":
    main()
