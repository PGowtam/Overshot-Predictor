"""
Iteration 2: Expanding Window Cross-Validation Tensor Builder

Builds 3 Folds of Train/Val/Test tensors based on expanding date windows.
Uses the execution-priced labels generated in Phase 9.
Features (snapshots, macro_vectors) are identical since they are price-invariant.

Folds Config:
Fold 1:
- Train:  2020-01-01 to 2021-12-31
- Val:    2022-01-01 to 2022-06-30
- Test:   2022-07-01 to 2022-12-31

Fold 2:
- Train:  2020-01-01 to 2022-06-30
- Val:    2022-07-01 to 2022-12-31
- Test:   2023-01-01 to 2023-06-30

Fold 3:
- Train:  2020-01-01 to 2022-12-31
- Val:    2023-01-01 to 2023-06-30
- Test:   2023-07-01 to 2023-12-31
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
EXEC_DIR = BASE_DIR / "outputs" / "exec"
# We reuse the original features and snapshots because they don't depend on pricing mode
FEATURE_DIR = EXEC_DIR / "features"
SNAPSHOT_DIR = FEATURE_DIR / "snapshots"
CV_DIR = EXEC_DIR / "cv"

CONTEXT_BRICKS = 10
FAST_BRICK_THRESHOLD = 2
CHAIN_DEPTH_THRESHOLD = 5
CHAIN_DURATION_THRESHOLD = 10

FOLDS = {
    1: {
        "train_start": pd.Timestamp("2020-01-01", tz="UTC"),
        "train_end": pd.Timestamp("2022-01-01", tz="UTC"),
        "val_end": pd.Timestamp("2022-07-01", tz="UTC"),
        "test_end": pd.Timestamp("2023-01-01", tz="UTC")
    },
    2: {
        "train_start": pd.Timestamp("2020-01-01", tz="UTC"),
        "train_end": pd.Timestamp("2022-07-01", tz="UTC"),
        "val_end": pd.Timestamp("2023-01-01", tz="UTC"),
        "test_end": pd.Timestamp("2023-07-01", tz="UTC")
    },
    3: {
        "train_start": pd.Timestamp("2020-01-01", tz="UTC"),
        "train_end": pd.Timestamp("2023-01-01", tz="UTC"),
        "val_end": pd.Timestamp("2023-07-01", tz="UTC"),
        "test_end": pd.Timestamp("2024-01-01", tz="UTC")
    }
}


def get_split(date: pd.Timestamp, fold_config: dict) -> str:
    if date < fold_config["train_start"]:
        return "ignore"
    elif date < fold_config["train_end"]:
        return "train"
    elif date < fold_config["val_end"]:
        return "val"
    elif date < fold_config["test_end"]:
        return "test"
    else:
        return "ignore"


def compute_chain_depths(durations: np.ndarray) -> np.ndarray:
    n = len(durations)
    depths = np.zeros(n, dtype=np.int32)
    for i in range(n):
        depth = 0
        for j in range(i - 1, -1, -1):
            if durations[j] < CHAIN_DURATION_THRESHOLD:
                depth += 1
            else:
                break
        depths[i] = depth
    return depths


def build_fold_tensors(fold_num: int, labels: pd.DataFrame, macro_vectors: np.ndarray, chain_depths: np.ndarray):
    print(f"\n=============================================")
    print(f" 🚀 Building Tensors for Fold {fold_num}")
    print(f"=============================================")
    
    fold_dir = CV_DIR / f"fold_{fold_num}" / "tensors"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    config = FOLDS[fold_num]
    n_bricks = len(labels)
    
    samples = {split: {"micro": [], "macro": [], "y_class": [], "y_mag": [], "weights": []}
               for split in ["train", "val", "test"]}
               
    skipped_context = 0
    skipped_exclude = 0
    skipped_train_fast = 0
    skipped_no_label = 0
    skipped_ignore = 0
    
    t_start = time.time()
    
    for i in range(n_bricks):
        row = labels.iloc[i]
        
        split = get_split(row["date"], config)
        if split == "ignore":
            skipped_ignore += 1
            continue
            
        if i < CONTEXT_BRICKS:
            skipped_context += 1
            continue
            
        if bool(row.get("exclude_flag", False)):
            skipped_exclude += 1
            continue
            
        if pd.isna(row.get("y_class")):
            skipped_no_label += 1
            continue
            
        duration = float(row.get("duration_seconds", 0))
        if split == "train" and duration < FAST_BRICK_THRESHOLD:
            skipped_train_fast += 1
            continue
            
        # Context building
        snapshot_stack = []
        try:
            for j in range(i - CONTEXT_BRICKS + 1, i + 1):
                snap_bid = int(labels.iloc[j]["brick_id"])
                snap_path = SNAPSHOT_DIR / f"snapshot_{snap_bid}.npy"
                snap = np.load(snap_path)
                snapshot_stack.append(snap)
            micro_tensor = np.stack(snapshot_stack)
        except Exception as e:
            continue
            
        macro_tensor = macro_vectors[i - CONTEXT_BRICKS + 1: i + 1]
        weight = 0.5 if (split == "train" and chain_depths[i] > CHAIN_DEPTH_THRESHOLD) else 1.0
        
        samples[split]["micro"].append(micro_tensor)
        samples[split]["macro"].append(macro_tensor)
        samples[split]["y_class"].append(float(row["y_class"]))
        samples[split]["y_mag"].append(float(row["y_mag"]))
        samples[split]["weights"].append(weight)
        
        if (i + 1) % 10000 == 0:
            print(f"  [{i+1:>6,}/{n_bricks:,}]")
            
    print(f"\n✅ Fold {fold_num} collection complete in {time.time() - t_start:.2f}s")
    print(f"   Ignored boundaries: {skipped_ignore:,}")
    print(f"   Excluded tags:      {skipped_exclude:,}")
    print(f"   Train Fast-skipped: {skipped_train_fast:,}")
    
    # Save the splits
    for split in ["train", "val", "test"]:
        n = len(samples[split]["micro"])
        print(f"   {split}: {n:,} samples")
        if n == 0: continue
        
        np.save(fold_dir / f"{split}_micro.npy", np.stack(samples[split]["micro"]))
        np.save(fold_dir / f"{split}_macro.npy", np.stack(samples[split]["macro"]))
        np.save(fold_dir / f"{split}_y_class.npy", np.array(samples[split]["y_class"], dtype=np.float32))
        np.save(fold_dir / f"{split}_y_mag.npy", np.array(samples[split]["y_mag"], dtype=np.float32))
        if split == "train":
            np.save(fold_dir / f"{split}_weights.npy", np.array(samples[split]["weights"], dtype=np.float32))


def main():
    print("=" * 60)
    print(" IT2 — K-Fold Tensor Builder (Execution Pricing)")
    print("=" * 60)
    
    # Load the execution-priced labels directly from outputs/exec
    labels_path = EXEC_DIR / "labels.parquet"
    if not labels_path.exists():
        print(f"❌ Execution Labels not found: {labels_path}")
        return
        
    try:
        labels = pd.read_parquet(labels_path)
    except Exception as e:
        print("Pyarrow failed reading parquet, regenerating execution labels inline...")
        import label_generator as lg
        CSV_PATH = BASE_DIR / "Data" / "Raw" / "renko_with_tick_outcomes_no_be_XAUUSD20-24.csv"
        TICK_DIR = BASE_DIR / "Data" / "Raw" / "Ticks"
        labels = lg.generate_all_labels(
            renko_csv_path=str(CSV_PATH),
            tick_dir=str(TICK_DIR),
            pricing_mode="execution"
        )
        
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    n_bricks = len(labels)
    print(f"Loaded {n_bricks:,} execution labels.")
    
    macro_path = FEATURE_DIR / "macro_vectors.npy"
    if not macro_path.exists():
        print(f"❌ Macro vectors not found: {macro_path}")
        return
    macro_vectors = np.load(macro_path)
    
    durations = labels["duration_seconds"].fillna(0).values
    chain_depths = compute_chain_depths(durations)
    
    for fold in [1, 2, 3]:
        build_fold_tensors(fold, labels, macro_vectors, chain_depths)

if __name__ == "__main__":
    main()
