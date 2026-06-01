"""
2RR Overshot: Tensor Construction
==================================
Builds training tensors for the 2-brick overshot prediction task.

The micro and macro tensors are IDENTICAL to the fallback pipeline —
only the classification label changes:
    y_class_2rr = 1.0  if  y_mag >= 2.0   (2-brick continuation)
    y_class_2rr = 0.0  if  y_mag <  2.0   (stopped out before 2 bricks)

Reads from:  outputs/fallback/features/snapshots/  (reuses fallback snapshots)
             outputs/fallback/features/macro_vectors.npy
             outputs/exec/labels.parquet
Writes to:   outputs/fallback_2rr/tensors/
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

# Output for 2RR experiment
RR2_DIR = OUTPUT_DIR / "fallback_2rr"
TENSOR_DIR = RR2_DIR / "tensors"

# Labels from exec pipeline
LABELS_PATH = OUTPUT_DIR / "exec" / "labels.parquet"

# ── Constants (same as fallback) ───────────────────────────────
CONTEXT_BRICKS = 10
FAST_BRICK_THRESHOLD = 2       # seconds — exclude from training
CHAIN_DEPTH_THRESHOLD = 5
CHAIN_DURATION_THRESHOLD = 10  # seconds

# ── 2RR THRESHOLD ─────────────────────────────────────────────
OVERSHOT_THRESHOLD = 2.0       # y_mag >= 2.0 → WIN

# ── Split date boundaries (UTC) ────────────────────────────────
SPLIT_DATES = {
    "val_start":     pd.Timestamp("2023-01-01", tz="UTC"),
    "test_start":    pd.Timestamp("2023-07-01", tz="UTC"),
    "holdout_start": pd.Timestamp("2024-01-01", tz="UTC"),
}


def assign_split(date: pd.Timestamp) -> str:
    if date < SPLIT_DATES["val_start"]:
        return "train"
    elif date < SPLIT_DATES["test_start"]:
        return "val"
    elif date < SPLIT_DATES["holdout_start"]:
        return "test"
    else:
        return "holdout"


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


def build_and_save_tensors():
    """Build tensors with 2RR labels from existing fallback snapshots."""
    # Load labels
    labels = pd.read_parquet(LABELS_PATH)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    n_bricks = len(labels)
    print(f"Loaded {n_bricks:,} bricks from exec labels")

    # Load macro-vectors from fallback feature engine
    macro_vectors = np.load(FEATURE_DIR / "macro_vectors.npy")
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

        if i < CONTEXT_BRICKS:
            skipped_context += 1
            continue

        if bool(row["exclude_flag"]):
            skipped_exclude += 1
            continue

        if pd.isna(row["y_mag"]):
            skipped_no_label += 1
            continue

        split = assign_split(row["date"])

        duration = float(row["duration_seconds"]) if pd.notna(row["duration_seconds"]) else 0.0
        if split == "train" and duration < FAST_BRICK_THRESHOLD:
            skipped_train_fast += 1
            continue

        # Load last 10 snapshots → (10, 100, 9)
        snapshot_stack = []
        for j in range(i - CONTEXT_BRICKS + 1, i + 1):
            snap_bid = int(labels.iloc[j]["brick_id"])
            snap_path = SNAPSHOT_DIR / f"snapshot_{snap_bid}.npy"
            snap = np.load(snap_path)
            snapshot_stack.append(snap)
        micro_tensor = np.stack(snapshot_stack)

        macro_tensor = macro_vectors[i - CONTEXT_BRICKS + 1: i + 1]

        weight = 0.5 if (split == "train" and chain_depths[i] > CHAIN_DEPTH_THRESHOLD) else 1.0

        # ── THE KEY CHANGE: 2RR label ──────────────────────────
        y_mag_val = float(row["y_mag"])
        y_class_2rr = 1.0 if y_mag_val >= OVERSHOT_THRESHOLD else 0.0

        samples[split]["micro"].append(micro_tensor)
        samples[split]["macro"].append(macro_tensor)
        samples[split]["y_class"].append(y_class_2rr)
        samples[split]["y_mag"].append(y_mag_val)
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
              f"2RR_WIN={win_count:,} 2RR_LOSS={loss_count:,} "
              f"(WR={split_info[split]['win_rate']:.3f})")

    elapsed = time.time() - t_start
    print(f"\n✅ 2RR Tensors built in {elapsed:.0f}s")
    print(f"   Skipped: context={skipped_context}, excluded={skipped_exclude}, "
          f"train_fast={skipped_train_fast}, no_label={skipped_no_label}")

    with open(TENSOR_DIR / "split_metadata.json", "w") as f:
        json.dump(split_info, f, indent=2, default=str)

    return split_info


def validate_tensors():
    with open(TENSOR_DIR / "split_metadata.json", "r") as f:
        split_info = json.load(f)

    print("\n" + "=" * 60)
    print(" 2RR Tensor Validation")
    print("=" * 60)

    all_pass = True

    # Shape check
    for split in ["train", "val", "test"]:
        micro_path = TENSOR_DIR / f"{split}_micro.npy"
        if not micro_path.exists():
            continue
        micro = np.load(micro_path)
        macro = np.load(TENSOR_DIR / f"{split}_macro.npy")
        y_class = np.load(TENSOR_DIR / f"{split}_y_class.npy")

        shape_ok = micro.shape[1:] == (10, 100, 9) and macro.shape == (micro.shape[0], 10, 3)
        print(f"  {split}: micro={micro.shape}, macro={macro.shape} {'✅' if shape_ok else '❌'}")
        all_pass &= shape_ok

        # Verify 2RR labels: y_class should be 1 iff y_mag >= 2.0
        y_mag = np.load(TENSOR_DIR / f"{split}_y_mag.npy")
        expected = (y_mag >= OVERSHOT_THRESHOLD).astype(np.float32)
        label_match = np.all(y_class == expected)
        print(f"  {split}: 2RR labels correct {'✅' if label_match else '❌'}")
        all_pass &= label_match

    # Print split sizes
    print(f"\n📊 2RR Split sizes:")
    for split in ["train", "val", "test", "holdout"]:
        info = split_info[split]
        n = info.get("n", 0)
        if n == 0:
            print(f"  {split:8s}: empty")
            continue
        wr = info["win_rate"]
        print(f"  {split:8s}: {n:>6,}  2RR_WIN={info['win']:<6,} 2RR_LOSS={info['loss']:<6,}  WR={wr:.3f}")

    print(f"\n{'ALL CHECKS PASSED ✅' if all_pass else 'SOME CHECKS FAILED ❌'}")
    return all_pass


def main():
    print("=" * 60)
    print(" 2RR Overshot: Tensor Construction")
    print(f" Label: y_class = 1 if y_mag >= {OVERSHOT_THRESHOLD}")
    print("=" * 60)

    if not LABELS_PATH.exists():
        print(f"ERROR: {LABELS_PATH} not found.")
        sys.exit(1)
    if not SNAPSHOT_DIR.exists():
        print("ERROR: Fallback snapshots not found. Run fallback pipeline first.")
        sys.exit(1)

    build_and_save_tensors()
    passed = validate_tensors()

    if not passed:
        sys.exit(1)

    print(f"\n💾 2RR Tensors saved to {TENSOR_DIR}")


if __name__ == "__main__":
    main()
