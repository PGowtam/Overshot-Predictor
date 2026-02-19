"""
Phase 3: Buffer Simulation (FR-BF-01, FR-BF-02, FR-BF-03)

Simulates the Micro-Buffer as it would operate in live trading:
- deque(maxlen=100) of 9D tick vectors
- NEVER reset between bricks (continuous)
- At each brick close, snapshot → (100, 9) with zero-padding at front
- Rewrites Flag_Curr (col 6) and Decay (col 8) relative to current brick

Output:
  outputs/features/snapshots/snapshot_{brick_id}.npy  — (100, 9) per brick
  outputs/features/buffer_metadata.parquet            — brick_id, n_real, n_padded
"""

import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
import sys
import time

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
SNAPSHOT_DIR = FEATURE_DIR / "snapshots"

# ── Feature column indices ─────────────────────────────────────
IDX_FLAG_CURR = 6   # 1 if tick belongs to current brick
IDX_DECAY     = 8   # (current_brick - tick_brick) / max_depth

# Buffer capacity
BUFFER_SIZE = 100


# ═══════════════════════════════════════════════════════════════
# 3.1  Micro-Buffer Simulation
# ═══════════════════════════════════════════════════════════════

def simulate_buffers(feature_dir: Path = None):
    """Simulate the micro-buffer across all bricks.

    For each brick:
      1. Load tick vectors from Phase 2
      2. Tag each tick with its originating brick_id
      3. Append to the continuous deque(maxlen=100)
      4. Snapshot at brick close → (100, 9)
      5. Rewrite Flag_Curr and Decay relative to current brick
      6. Zero-pad at front if < 100 ticks

    Returns:
        metadata_rows (list[dict]), nan_count (int)
    """
    if feature_dir is None:
        feature_dir = FEATURE_DIR

    # Load brick metadata from Phase 2
    meta_path = feature_dir / "brick_metadata.parquet"
    if not meta_path.exists():
        print("ERROR: brick_metadata.parquet not found. Run Phase 2 first.")
        sys.exit(1)

    meta = pd.read_parquet(meta_path)
    n_bricks = len(meta)
    print(f"Loaded metadata for {n_bricks:,} bricks")

    # Create snapshot dir
    snapshot_dir = feature_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # The micro-buffer stores (vector, origin_brick_idx) tuples
    # We need origin_brick_idx to compute Flag_Curr and Decay
    micro_buffer = deque(maxlen=BUFFER_SIZE)

    # Tracking
    metadata_rows = []
    nan_count = 0
    t_start = time.time()

    for brick_idx in range(n_bricks):
        brick_id = int(meta.iloc[brick_idx]["brick_id"])
        n_ticks_this_brick = int(meta.iloc[brick_idx]["n_ticks"])

        # Load tick vectors from Phase 2
        vec_path = feature_dir / f"tick_vectors_{brick_id}.npy"
        if vec_path.exists():
            tick_vectors = np.load(vec_path)  # (N, 9)
        else:
            tick_vectors = np.zeros((0, 9), dtype=np.float32)

        # Append each tick to the continuous buffer with its brick index
        for t in range(len(tick_vectors)):
            micro_buffer.append((tick_vectors[t].copy(), brick_idx))

        # ── Snapshot at brick close ─────────────────────────────
        buf_len = len(micro_buffer)

        if buf_len == 0:
            # No ticks at all yet
            snapshot = np.zeros((BUFFER_SIZE, 9), dtype=np.float32)
            n_real = 0
        else:
            # Build snapshot from buffer contents
            snapshot = np.zeros((BUFFER_SIZE, 9), dtype=np.float32)
            n_real = buf_len

            # Fill from the end (newest at bottom)
            start_pos = BUFFER_SIZE - buf_len
            for j, (vec, origin_idx) in enumerate(micro_buffer):
                row = start_pos + j
                snapshot[row] = vec

                # Rewrite Flag_Curr: 1 if origin == current brick, else 0
                snapshot[row, IDX_FLAG_CURR] = 1.0 if origin_idx == brick_idx else 0.0

                # Rewrite Decay: (current_brick - origin_brick) / max_depth
                # max_depth = BUFFER_SIZE (normalise to [0, 1])
                brick_distance = brick_idx - origin_idx
                snapshot[row, IDX_DECAY] = min(brick_distance / max(BUFFER_SIZE, 1), 1.0)

        n_padded = BUFFER_SIZE - n_real

        # Check for NaN
        if np.any(np.isnan(snapshot)):
            nan_count += 1

        # Save snapshot
        np.save(snapshot_dir / f"snapshot_{brick_id}.npy", snapshot)

        metadata_rows.append({
            "brick_id": brick_id,
            "n_real_ticks": n_real,
            "n_padded": n_padded,
            "n_curr_brick_ticks": n_ticks_this_brick,
        })

        # Progress
        if (brick_idx + 1) % 5000 == 0 or brick_idx == n_bricks - 1:
            elapsed = time.time() - t_start
            rate = (brick_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_bricks - brick_idx - 1) / rate if rate > 0 else 0
            print(f"  [{brick_idx+1:>6,}/{n_bricks:,}] ({(brick_idx+1)/n_bricks*100:.1f}%) "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    # Save metadata
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_parquet(feature_dir / "buffer_metadata.parquet", index=False)

    elapsed_total = time.time() - t_start
    print(f"\n✅ Done in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"   Snapshots saved: {n_bricks:,}")
    print(f"   NaN snapshots: {nan_count}")

    return metadata_rows, nan_count


# ═══════════════════════════════════════════════════════════════
# 3.3  Validation
# ═══════════════════════════════════════════════════════════════

def validate_buffers(feature_dir: Path = None):
    """Validate buffer simulation outputs."""
    if feature_dir is None:
        feature_dir = FEATURE_DIR

    snapshot_dir = feature_dir / "snapshots"
    meta = pd.read_parquet(feature_dir / "buffer_metadata.parquet")
    brick_meta = pd.read_parquet(feature_dir / "brick_metadata.parquet")
    n_bricks = len(meta)

    print("\n" + "=" * 60)
    print(" Phase 3 Validation")
    print("=" * 60)

    all_pass = True

    # ── Check 1: All snapshots shape (100, 9) ───────────────────
    shape_errors = 0
    for _, row in meta.iterrows():
        bid = int(row["brick_id"])
        path = snapshot_dir / f"snapshot_{bid}.npy"
        if path.exists():
            s = np.load(path)
            if s.shape != (BUFFER_SIZE, 9):
                shape_errors += 1
        else:
            shape_errors += 1
    check1 = shape_errors == 0
    print(f"\n✓ All snapshots shape (100,9): {'PASS ✅' if check1 else f'FAIL ❌ ({shape_errors} errors)'}")
    all_pass &= check1

    # ── Check 2: No NaN in any snapshot ─────────────────────────
    nan_snapshots = 0
    for _, row in meta.iterrows():
        bid = int(row["brick_id"])
        s = np.load(snapshot_dir / f"snapshot_{bid}.npy")
        if np.any(np.isnan(s)):
            nan_snapshots += 1
    check2 = nan_snapshots == 0
    print(f"✓ No NaN in snapshots: {'PASS ✅' if check2 else f'FAIL ❌ ({nan_snapshots} found)'}")
    all_pass &= check2

    # ── Check 3: Flag_Curr count for fast bricks ────────────────
    # For fast bricks (few ticks), Flag_Curr=1 count should match n_ticks_this_brick
    print(f"\n📊 Flag_Curr check (5 fast bricks, n_ticks < 50):")
    fast_bricks = meta[meta["n_curr_brick_ticks"].between(1, 50)].head(5)
    flag_ok = True
    for _, row in fast_bricks.iterrows():
        bid = int(row["brick_id"])
        expected_curr = int(row["n_curr_brick_ticks"])
        s = np.load(snapshot_dir / f"snapshot_{bid}.npy")
        actual_curr = int(np.sum(s[:, IDX_FLAG_CURR] == 1.0))
        match = actual_curr == expected_curr
        flag_ok &= match
        print(f"  brick_id={bid}: expected Flag_Curr=1 count={expected_curr}, "
              f"actual={actual_curr} {'✅' if match else '❌'}")
    all_pass &= flag_ok

    # ── Check 4: Buffer continuity ──────────────────────────────
    print(f"\n📊 Buffer continuity (10 consecutive pairs):")
    continuity_ok = True
    # Pick 10 pairs starting from brick 100 (buffer should be full by then)
    start_idx = min(100, n_bricks - 11)
    for i in range(start_idx, start_idx + 10):
        if i + 1 >= n_bricks:
            break
        bid_i = int(meta.iloc[i]["brick_id"])
        bid_i1 = int(meta.iloc[i + 1]["brick_id"])
        n_ticks_next = int(meta.iloc[i + 1]["n_curr_brick_ticks"])

        snap_i = np.load(snapshot_dir / f"snapshot_{bid_i}.npy")
        snap_i1 = np.load(snapshot_dir / f"snapshot_{bid_i1}.npy")

        if n_ticks_next >= BUFFER_SIZE:
            # Next brick filled the entire buffer — no overlap expected
            print(f"  bricks {bid_i}→{bid_i1}: next has {n_ticks_next} ticks (fills buffer), skip")
            continue

        # The overlap region: in snap_i1, the first (100 - n_ticks_next) rows
        # should match the last (100 - n_ticks_next) rows of snap_i
        # BUT Flag_Curr and Decay will differ (rewritten), so compare only cols 0-5, 7
        overlap = BUFFER_SIZE - n_ticks_next
        if overlap <= 0:
            continue

        # Compare z-scored features (cols 0-5) and Flag_Zone (col 7)
        compare_cols = [0, 1, 2, 3, 4, 5, 7]
        tail_i = snap_i[-overlap:, :][:, compare_cols]
        head_i1 = snap_i1[:overlap, :][:, compare_cols]

        match = np.allclose(tail_i, head_i1, atol=1e-5)
        continuity_ok &= match
        print(f"  bricks {bid_i}→{bid_i1}: overlap={overlap}, "
              f"match={'✅' if match else '❌'}")

    all_pass &= continuity_ok

    # ── Check 5: Snapshot file count ────────────────────────────
    snap_files = list(snapshot_dir.glob("snapshot_*.npy"))
    check5 = len(snap_files) == n_bricks
    print(f"\n✓ Snapshot files: {len(snap_files)}/{n_bricks} "
          f"{'PASS ✅' if check5 else 'FAIL ❌'}")
    all_pass &= check5

    # ── Summary stats ───────────────────────────────────────────
    print(f"\n📊 Buffer fill statistics:")
    print(f"  Mean real ticks per snapshot: {meta['n_real_ticks'].mean():.1f}")
    print(f"  Mean padded zeros: {meta['n_padded'].mean():.1f}")
    print(f"  Bricks with full buffer (100 real): {(meta['n_real_ticks'] == BUFFER_SIZE).sum():,}")
    print(f"  Bricks with partial buffer: {(meta['n_real_ticks'] < BUFFER_SIZE).sum():,}")

    print("\n" + "=" * 60)
    print(f" {'ALL CHECKS PASSED ✅' if all_pass else 'SOME CHECKS FAILED ❌'}")
    print("=" * 60)

    return all_pass


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Phase 3: Buffer Simulation")
    print("=" * 60)

    metadata_rows, nan_count = simulate_buffers()
    passed = validate_buffers()

    if not passed:
        print("\n⚠️  Some validation checks failed. Review output above.")
        sys.exit(1)

    print(f"\n💾 Snapshots saved to {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
