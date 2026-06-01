"""
Volume Fallback: Tensor Construction
=====================================
Assembles final training tensors from fallback snapshots and macro-vectors.

Identical logic to tensor_builder.py but reads from outputs/fallback/features/
and writes to outputs/fallback/tensors/.

Uses exec labels for split assignments, exclusions, and sample weights.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import time

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
FALLBACK_DIR = OUTPUT_DIR / "fallback"
FEATURE_DIR = FALLBACK_DIR / "features"
SNAPSHOT_DIR = FEATURE_DIR / "snapshots"
TENSOR_DIR = FALLBACK_DIR / "tensors"

# Labels from exec pipeline
LABELS_PATH = OUTPUT_DIR / "exec" / "labels.parquet"

# ── Constants ──────────────────────────────────────────────────
CONTEXT_BRICKS = 10
FAST_BRICK_THRESHOLD = 2       # seconds — exclude from training
CHAIN_DEPTH_THRESHOLD = 5
CHAIN_DURATION_THRESHOLD = 10  # seconds — counts as "fast" for chain depth

# ── Split date boundaries (UTC) ────────────────────────────────
SPLIT_DATES = {
    "val_start":     pd.Timestamp("2023-01-01", tz="UTC"),
    "test_start":    pd.Timestamp("2023-07-01", tz="UTC"),
    "holdout_start": pd.Timestamp("2024-01-01", tz="UTC"),
}


def assign_split(date: pd.Timestamp) -> str:
    """Assign a brick to a walk-forward split based on its date."""
    if date < SPLIT_DATES["val_start"]:
        return "train"
    elif date < SPLIT_DATES["test_start"]:
        return "val"
    elif date < SPLIT_DATES["holdout_start"]:
        return "test"
    else:
        return "holdout"


def compute_chain_depths(durations: np.ndarray) -> np.ndarray:
    """Compute fast-brick chain depth for each brick."""
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


def build_and_save_tensors():
    """Build tensors from fallback snapshots and macro-vectors.

    Returns:
        split_info (dict): metadata about each split
    """
    # Load labels
    labels = pd.read_parquet(LABELS_PATH)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    n_bricks = len(labels)
    print(f"Loaded {n_bricks:,} bricks from exec labels")

    # Load macro-vectors from fallback feature engine
    macro_vectors = np.load(FEATURE_DIR / "macro_vectors.npy")  # (n_bricks, 3)
    print(f"Loaded macro vectors: {macro_vectors.shape}")

    # Compute chain depths
    durations = labels["duration_seconds"].fillna(0).values
    chain_depths = compute_chain_depths(durations)

    # Create output dir
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── Collect valid samples ──────────────────────────────────
    samples = {split: {"micro": [], "macro": [], "y_class": [], "y_mag": [],
                        "weights": [], "brick_ids": [], "dates": []}
               for split in ["train", "val", "test", "holdout"]}

    skipped_context = 0
    skipped_exclude = 0
    skipped_train_fast = 0
    skipped_no_label = 0

    for i in range(n_bricks):
        row = labels.iloc[i]
        brick_id = int(row["brick_id"])

        # Need 10 bricks of context
        if i < CONTEXT_BRICKS:
            skipped_context += 1
            continue

        # Excluded bricks
        if bool(row["exclude_flag"]):
            skipped_exclude += 1
            continue

        # Must have valid y_class
        if pd.isna(row["y_class"]):
            skipped_no_label += 1
            continue

        # Assign split
        split = assign_split(row["date"])

        # Training exclusion: duration < 2s
        duration = float(row["duration_seconds"]) if pd.notna(row["duration_seconds"]) else 0.0
        if split == "train" and duration < FAST_BRICK_THRESHOLD:
            skipped_train_fast += 1
            continue

        # Load last 10 snapshots → (10, 100, 9)
        snapshot_stack = []
        for j in range(i - CONTEXT_BRICKS + 1, i + 1):
            snap_bid = int(labels.iloc[j]["brick_id"])
            snap_path = SNAPSHOT_DIR / f"snapshot_{snap_bid}.npy"
            snap = np.load(snap_path)  # (100, 9)
            snapshot_stack.append(snap)
        micro_tensor = np.stack(snapshot_stack)  # (10, 100, 9)

        # Stack last 10 macro-vectors → (10, 3)
        macro_tensor = macro_vectors[i - CONTEXT_BRICKS + 1: i + 1]  # (10, 3)

        # Sample weight
        weight = 0.5 if (split == "train" and chain_depths[i] > CHAIN_DEPTH_THRESHOLD) else 1.0

        # Store
        samples[split]["micro"].append(micro_tensor)
        samples[split]["macro"].append(macro_tensor)
        samples[split]["y_class"].append(float(row["y_class"]))
        samples[split]["y_mag"].append(float(row["y_mag"]))
        samples[split]["weights"].append(weight)
        samples[split]["brick_ids"].append(brick_id)
        samples[split]["dates"].append(row["date"])

        if (i + 1) % 5000 == 0:
            print(f"  [{i+1:>6,}/{n_bricks:,}] ({(i+1)/n_bricks*100:.1f}%)")

    # ── Stack and save ─────────────────────────────────────────
    split_info = {}
    for split in ["train", "val", "test", "holdout"]:
        n = len(samples[split]["micro"])
        if n == 0:
            split_info[split] = {"n": 0, "win_rate": None, "date_range": None}
            continue

        micro = np.stack(samples[split]["micro"])
        macro = np.stack(samples[split]["macro"])
        y_class = np.array(samples[split]["y_class"], dtype=np.float32)
        y_mag = np.array(samples[split]["y_mag"], dtype=np.float32)

        np.save(TENSOR_DIR / f"{split}_micro.npy", micro)
        np.save(TENSOR_DIR / f"{split}_macro.npy", macro)
        np.save(TENSOR_DIR / f"{split}_y_class.npy", y_class)
        np.save(TENSOR_DIR / f"{split}_y_mag.npy", y_mag)

        if split == "train":
            weights = np.array(samples[split]["weights"], dtype=np.float32)
            np.save(TENSOR_DIR / "train_weights.npy", weights)

        # Split metadata
        dates = samples[split]["dates"]
        win_count = int(np.sum(y_class == 1.0))
        loss_count = int(np.sum(y_class == 0.0))
        split_info[split] = {
            "n": n,
            "win": win_count,
            "loss": loss_count,
            "win_rate": round(win_count / n, 4) if n > 0 else None,
            "date_min": str(min(dates)),
            "date_max": str(max(dates)),
        }
        if split == "train":
            split_info[split]["n_downweighted"] = int(np.sum(
                np.array(samples[split]["weights"]) == 0.5))

        print(f"  {split:8s}: {n:>6,} samples, "
              f"WIN={win_count:,} LOSS={loss_count:,} "
              f"({split_info[split]['win_rate']:.3f})")

    elapsed = time.time() - t_start
    print(f"\n✅ Tensors built in {elapsed:.0f}s")
    print(f"   Skipped: context={skipped_context}, excluded={skipped_exclude}, "
          f"train_fast={skipped_train_fast}, no_label={skipped_no_label}")

    # Save split metadata
    with open(TENSOR_DIR / "split_metadata.json", "w") as f:
        json.dump(split_info, f, indent=2, default=str)

    return split_info


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════

def validate_tensors():
    """Validate fallback tensor construction outputs."""
    with open(TENSOR_DIR / "split_metadata.json", "r") as f:
        split_info = json.load(f)

    print("\n" + "=" * 60)
    print(" Fallback Tensor Validation")
    print("=" * 60)

    all_pass = True

    # Check 1: Tensor shapes
    shape_ok = True
    for split in ["train", "val", "test", "holdout"]:
        micro_path = TENSOR_DIR / f"{split}_micro.npy"
        if not micro_path.exists():
            continue
        micro = np.load(micro_path)
        macro = np.load(TENSOR_DIR / f"{split}_macro.npy")
        n = micro.shape[0]
        if micro.shape[1:] != (10, 100, 9):
            print(f"  ❌ {split}_micro shape: {micro.shape}")
            shape_ok = False
        if macro.shape != (n, 10, 3):
            print(f"  ❌ {split}_macro shape: {macro.shape}")
            shape_ok = False
    print(f"\n✓ Tensor shapes correct: {'PASS ✅' if shape_ok else 'FAIL ❌'}")
    all_pass &= shape_ok

    # Check 2: No NaN
    nan_ok = True
    for split in ["train", "val", "test", "holdout"]:
        for suffix in ["micro", "macro", "y_class", "y_mag"]:
            path = TENSOR_DIR / f"{split}_{suffix}.npy"
            if path.exists():
                arr = np.load(path)
                if np.any(np.isnan(arr)):
                    print(f"  ❌ NaN in {split}_{suffix}")
                    nan_ok = False
    print(f"✓ No NaN in tensors: {'PASS ✅' if nan_ok else 'FAIL ❌'}")
    all_pass &= nan_ok

    # Check 3: z_Depth (col 1) and z_Susc (col 2) should be 0.0 in micro tensors
    zero_ok = True
    for split in ["train", "val", "test"]:
        micro_path = TENSOR_DIR / f"{split}_micro.npy"
        if micro_path.exists():
            micro = np.load(micro_path)
            # Check real (non-padded) portion
            max_depth = np.max(np.abs(micro[:, :, :, 1]))
            max_susc = np.max(np.abs(micro[:, :, :, 2]))
            if max_depth > 1e-10 or max_susc > 1e-10:
                print(f"  ❌ {split} has non-zero z_Depth/z_Susc: "
                      f"max|z_Depth|={max_depth:.2e}, max|z_Susc|={max_susc:.2e}")
                zero_ok = False
    print(f"✓ z_Depth ≡ 0, z_Susc ≡ 0 in tensors: {'PASS ✅' if zero_ok else 'FAIL ❌'}")
    all_pass &= zero_ok

    # Print split info
    print(f"\n📊 Split sizes:")
    for split in ["train", "val", "test", "holdout"]:
        info = split_info[split]
        n = info["n"] if info["n"] else 0
        if n == 0:
            print(f"  {split:8s}: empty")
            continue
        wr = info["win_rate"]
        extra = ""
        if split == "train" and "n_downweighted" in info:
            extra = f", downweighted={info['n_downweighted']:,}"
        print(f"  {split:8s}: {n:>6,}  WIN={info['win']:<6,} LOSS={info['loss']:<6,}  "
              f"WR={wr:.3f}{extra}")

    print("\n" + "=" * 60)
    print(f" {'ALL CHECKS PASSED ✅' if all_pass else 'SOME CHECKS FAILED ❌'}")
    print("=" * 60)

    return all_pass


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Volume Fallback: Tensor Construction")
    print("=" * 60)

    if not LABELS_PATH.exists():
        print(f"ERROR: {LABELS_PATH} not found.")
        sys.exit(1)
    if not (FEATURE_DIR / "macro_vectors.npy").exists():
        print("ERROR: macro_vectors.npy not found. Run feature_engine_fallback.py first.")
        sys.exit(1)
    if not SNAPSHOT_DIR.exists():
        print("ERROR: snapshots dir not found. Run buffer_sim_fallback.py first.")
        sys.exit(1)

    split_info = build_and_save_tensors()
    passed = validate_tensors()

    if not passed:
        print("\n⚠️  Some validation checks failed. Review output above.")
        sys.exit(1)

    print(f"\n💾 Tensors saved to {TENSOR_DIR}")


if __name__ == "__main__":
    main()
