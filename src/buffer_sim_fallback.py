"""
Volume Fallback: Buffer Simulation
===================================
Simulates the Micro-Buffer for fallback features (FR-BF-01, FR-BF-02, FR-BF-03).

Identical logic to buffer_sim.py but reads from outputs/fallback/features/
and writes snapshots to outputs/fallback/features/snapshots/.
"""

import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
import sys
import time

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FALLBACK_DIR = BASE_DIR / "outputs" / "fallback"
FEATURE_DIR = FALLBACK_DIR / "features"
SNAPSHOT_DIR = FEATURE_DIR / "snapshots"

# ── Feature column indices ─────────────────────────────────────
IDX_FLAG_CURR = 6   # 1 if tick belongs to current brick
IDX_DECAY     = 8   # (current_brick - tick_brick) / max_depth

# Buffer capacity
BUFFER_SIZE = 100


# ═══════════════════════════════════════════════════════════════
# Micro-Buffer Simulation
# ═══════════════════════════════════════════════════════════════

def simulate_buffers():
    """Simulate the micro-buffer across all bricks using fallback features.

    For each brick:
      1. Load tick vectors from fallback feature engine
      2. Tag each tick with its originating brick_id
      3. Append to the continuous deque(maxlen=100)
      4. Snapshot at brick close → (100, 9)
      5. Rewrite Flag_Curr and Decay relative to current brick
      6. Zero-pad at front if < 100 ticks

    Returns:
        metadata_rows (list[dict]), nan_count (int)
    """
    # Load brick metadata from fallback feature engine
    meta_path = FEATURE_DIR / "brick_metadata.parquet"
    if not meta_path.exists():
        print("ERROR: brick_metadata.parquet not found. Run feature_engine_fallback.py first.")
        sys.exit(1)

    meta = pd.read_parquet(meta_path)
    n_bricks = len(meta)
    print(f"Loaded metadata for {n_bricks:,} bricks")

    # Create snapshot dir
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # The micro-buffer stores (vector, origin_brick_idx) tuples
    micro_buffer = deque(maxlen=BUFFER_SIZE)

    # Tracking
    metadata_rows = []
    nan_count = 0
    t_start = time.time()

    for brick_idx in range(n_bricks):
        brick_id = int(meta.iloc[brick_idx]["brick_id"])
        n_ticks_this_brick = int(meta.iloc[brick_idx]["n_ticks"])

        # Load tick vectors from fallback feature engine
        vec_path = FEATURE_DIR / f"tick_vectors_{brick_id}.npy"
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
            snapshot = np.zeros((BUFFER_SIZE, 9), dtype=np.float32)
            n_real = 0
        else:
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
                brick_distance = brick_idx - origin_idx
                snapshot[row, IDX_DECAY] = min(brick_distance / max(BUFFER_SIZE, 1), 1.0)

        n_padded = BUFFER_SIZE - n_real

        # Check for NaN
        if np.any(np.isnan(snapshot)):
            nan_count += 1

        # Save snapshot
        np.save(SNAPSHOT_DIR / f"snapshot_{brick_id}.npy", snapshot)

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
    meta_df.to_parquet(FEATURE_DIR / "buffer_metadata.parquet", index=False)

    elapsed_total = time.time() - t_start
    print(f"\n✅ Done in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"   Snapshots saved: {n_bricks:,}")
    print(f"   NaN snapshots: {nan_count}")

    return metadata_rows, nan_count


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════

def validate_buffers():
    """Validate buffer simulation outputs."""
    meta = pd.read_parquet(FEATURE_DIR / "buffer_metadata.parquet")
    n_bricks = len(meta)

    print("\n" + "=" * 60)
    print(" Fallback Buffer Validation")
    print("=" * 60)

    all_pass = True

    # Check 1: All snapshots shape (100, 9)
    shape_errors = 0
    for _, row in meta.iterrows():
        bid = int(row["brick_id"])
        path = SNAPSHOT_DIR / f"snapshot_{bid}.npy"
        if path.exists():
            s = np.load(path)
            if s.shape != (BUFFER_SIZE, 9):
                shape_errors += 1
        else:
            shape_errors += 1
    check1 = shape_errors == 0
    print(f"\n✓ All snapshots shape (100,9): {'PASS ✅' if check1 else f'FAIL ❌ ({shape_errors} errors)'}")
    all_pass &= check1

    # Check 2: No NaN
    nan_snapshots = 0
    for _, row in meta.iterrows():
        bid = int(row["brick_id"])
        s = np.load(SNAPSHOT_DIR / f"snapshot_{bid}.npy")
        if np.any(np.isnan(s)):
            nan_snapshots += 1
    check2 = nan_snapshots == 0
    print(f"✓ No NaN in snapshots: {'PASS ✅' if check2 else f'FAIL ❌ ({nan_snapshots} found)'}")
    all_pass &= check2

    # Check 3: z_Depth (col 1) and z_Susc (col 2) should be 0.0 everywhere
    depth_susc_ok = True
    sample_bricks = meta.sample(min(50, n_bricks), random_state=42)
    for _, row in sample_bricks.iterrows():
        bid = int(row["brick_id"])
        s = np.load(SNAPSHOT_DIR / f"snapshot_{bid}.npy")
        n_real = int(row["n_real_ticks"])
        if n_real > 0:
            real_part = s[BUFFER_SIZE - n_real:]
            if np.any(np.abs(real_part[:, 1]) > 1e-10) or np.any(np.abs(real_part[:, 2]) > 1e-10):
                depth_susc_ok = False
                break
    print(f"✓ z_Depth ≡ 0 and z_Susc ≡ 0 in snapshots: {'PASS ✅' if depth_susc_ok else 'FAIL ❌'}")
    all_pass &= depth_susc_ok

    # Check 4: Snapshot file count
    snap_files = list(SNAPSHOT_DIR.glob("snapshot_*.npy"))
    check4 = len(snap_files) == n_bricks
    print(f"✓ Snapshot files: {len(snap_files)}/{n_bricks} "
          f"{'PASS ✅' if check4 else 'FAIL ❌'}")
    all_pass &= check4

    # Summary stats
    print(f"\n📊 Buffer fill statistics:")
    print(f"  Mean real ticks per snapshot: {meta['n_real_ticks'].mean():.1f}")
    print(f"  Mean padded zeros: {meta['n_padded'].mean():.1f}")
    print(f"  Bricks with full buffer (100 real): {(meta['n_real_ticks'] == BUFFER_SIZE).sum():,}")

    print("\n" + "=" * 60)
    print(f" {'ALL CHECKS PASSED ✅' if all_pass else 'SOME CHECKS FAILED ❌'}")
    print("=" * 60)

    return all_pass


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Volume Fallback: Buffer Simulation")
    print("=" * 60)

    metadata_rows, nan_count = simulate_buffers()
    passed = validate_buffers()

    if not passed:
        print("\n⚠️  Some validation checks failed. Review output above.")
        sys.exit(1)

    print(f"\n💾 Snapshots saved to {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
