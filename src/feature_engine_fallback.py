"""
Volume Fallback Feature Engine
==============================
Computes the 9D tick feature vector and 3D macro-vector for every brick,
using the EXACT Volume Fallback Proxy from BrickOfTicks_Trader:

  raw_ofi   = sign(mid_k - mid_{k-1})   (tick direction)
  depth_raw = 0.0                         (no volume info)
  susc_raw  = 0.0                         (no volume info)

All other features (z_Vel, z_Spread, Progress, Flag_Curr, Flag_Zone, Decay)
are computed identically to the original feature_engine.py.

Reads labels from:  outputs/exec/labels.parquet
Outputs to:         outputs/fallback/features/
"""

import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from math import sqrt, log
import sys
import time

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "Raw"
TICK_DIR = DATA_DIR / "Ticks"
OUTPUT_DIR = BASE_DIR / "outputs"

# Input: exec labels (initial brick sizes)
LABELS_PATH = OUTPUT_DIR / "exec" / "labels.parquet"

# Output: fallback features
FALLBACK_DIR = OUTPUT_DIR / "fallback"
FEATURE_DIR = FALLBACK_DIR / "features"


# ═══════════════════════════════════════════════════════════════
# Rolling Z-Score (identical to original)
# ═══════════════════════════════════════════════════════════════

class RollingZScore:
    """O(1) incremental z-score with sliding window.

    Uses Welford-like incremental formula for full window:
        μ_new = μ_old + (x_new - x_old) / N
        M2_new = M2_old + (x_new - x_old) * ((x_new - μ_new) + (x_old - μ_old))

    Returns 0.0 when window has fewer than 30 values.
    Returns 0.0 when σ = 0 (constant values).
    """

    def __init__(self, window: int = 1000):
        self.window = window
        self.deque = deque(maxlen=window)
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x_new: float) -> float:
        N = len(self.deque)

        if N == self.window:
            # Full window — O(1) incremental update
            x_old = self.deque[0]  # About to be evicted
            self.deque.append(x_new)
            mean_new = self.mean + (x_new - x_old) / N
            self.M2 = self.M2 + (x_new - x_old) * ((x_new - mean_new) + (x_old - self.mean))
            self.mean = mean_new
            # Clamp M2 to avoid negative due to float errors
            if self.M2 < 0:
                self.M2 = 0.0
            sigma = sqrt(self.M2 / (N - 1)) if N > 1 else 0.0
            if sigma < 1e-12:
                return 0.0
            return (x_new - self.mean) / sigma
        else:
            # Filling phase
            self.deque.append(x_new)
            N = len(self.deque)
            if N < 30:
                return 0.0  # Not enough data
            # Recompute from scratch (small N, acceptable)
            arr = list(self.deque)
            self.mean = sum(arr) / N
            self.M2 = sum((x - self.mean) ** 2 for x in arr)
            sigma = sqrt(self.M2 / (N - 1)) if N > 1 else 0.0
            if sigma < 1e-12:
                return 0.0
            return (x_new - self.mean) / sigma


# ═══════════════════════════════════════════════════════════════
# Raw Feature Computations
# ═══════════════════════════════════════════════════════════════

def compute_velocity(t_k_ms: float, t_km1_ms: float) -> float:
    """Compute tick velocity: 1 / (dt_ms + 1e-3)."""
    dt = t_k_ms - t_km1_ms
    return 1.0 / (dt + 1e-3)


def compute_spread(ask: float, bid: float) -> float:
    """Compute spread."""
    return ask - bid


def compute_progress(mid: float, brick_open: float, brick_size: float) -> float:
    """Compute progress: (mid - brick_open) / brick_size."""
    return (mid - brick_open) / brick_size


def compute_flag_zone(mid: float, prev_brick_open: float, prev_brick_size: float) -> int:
    """1 if mid is beyond the previous brick's boundary."""
    return 1 if abs(mid - prev_brick_open) >= prev_brick_size else 0


def compute_macro_vector(duration_s: float, is_uptrend: bool,
                         brick_size: float, brick_size_history: list) -> np.ndarray:
    """Compute 3D macro-vector for a brick.

    [log(duration + 1), direction(±1), z_size]
    z_size = (brick_size - mean_50) / std_50 using last 50 brick sizes.
    """
    log_dur = log(duration_s + 1)
    direction = 1.0 if is_uptrend else -1.0

    if len(brick_size_history) < 2:
        z_size = 0.0
    else:
        recent = brick_size_history[-50:]
        mu = np.mean(recent)
        sigma = np.std(recent, ddof=1) if len(recent) > 1 else 0.0
        if sigma < 1e-12:
            z_size = 0.0
        else:
            z_size = (brick_size - mu) / sigma

    return np.array([log_dur, direction, z_size], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# Tick Loading
# ═══════════════════════════════════════════════════════════════

def _load_ticks_range(start_time: pd.Timestamp, end_time: pd.Timestamp,
                      cache: dict, tick_dir: Path) -> pd.DataFrame:
    """Load ticks between start_time and end_time, spanning multiple days if needed."""
    start_date = start_time.normalize()
    end_date = end_time.normalize()
    dates = pd.date_range(start_date, end_date, freq="D")

    frames = []
    for d in dates:
        key = (d.year, d.month, d.day)
        if key not in cache:
            path = tick_dir / str(d.year) / f"{d.month:02d}" / f"{d.day:02d}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
                cache[key] = df
            else:
                cache[key] = None  # Mark as missing
        cached = cache[key]
        if cached is not None and len(cached) > 0:
            frames.append(cached)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "bid", "bid_vol", "ask", "ask_vol"])

    if len(frames) == 1:
        all_ticks = frames[0]
    else:
        all_ticks = pd.concat(frames, ignore_index=True)
    mask = (all_ticks["timestamp"] > start_time) & (all_ticks["timestamp"] <= end_time)
    result = all_ticks[mask].reset_index(drop=True)
    return result


# ═══════════════════════════════════════════════════════════════
# Full Pipeline — Volume Fallback
# ═══════════════════════════════════════════════════════════════

def process_all_ticks_fallback():
    """Process all ticks using the Volume Fallback Proxy.

    For EVERY tick, regardless of actual volume data:
      raw_ofi   = sign(mid_k - mid_{k-1})
      depth_raw = 0.0
      susc_raw  = 0.0

    This simulates the model operating as if volume data was NEVER available.

    Outputs:
        - outputs/fallback/features/tick_vectors_{brick_id}.npy per brick
        - outputs/fallback/features/macro_vectors.npy
        - outputs/fallback/features/brick_metadata.parquet
    """
    # Load labels
    labels = pd.read_parquet(LABELS_PATH)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    n_bricks = len(labels)
    print(f"Loaded {n_bricks:,} bricks from exec labels")

    # Create output dirs
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize rolling z-score instances (never reset across bricks)
    # Note: z_OFI will see {-1, 0, +1} values only
    # z_Depth will see only 0.0 → will always return 0.0
    # z_Susc will see only 0.0 → will always return 0.0
    zs_ofi = RollingZScore(1000)
    zs_depth = RollingZScore(1000)
    zs_susc = RollingZScore(1000)
    zs_vel = RollingZScore(1000)
    zs_spread = RollingZScore(1000)

    # Macro-vector storage
    macro_vectors = np.zeros((n_bricks, 3), dtype=np.float32)
    brick_size_history = []

    # Metadata storage
    metadata_rows = []

    # Tick cache (day → DataFrame)
    _tick_cache = {}
    MAX_CACHE_SIZE = 5

    # Feature stats accumulators
    feat_stats = {i: {"sum": 0.0, "sum2": 0.0, "n": 0, "min": float("inf"), "max": float("-inf")}
                  for i in range(9)}
    total_ticks_processed = 0
    nan_inf_count = 0

    t_start = time.time()
    prev_brick_close = None

    for brick_idx in range(n_bricks):
        row = labels.iloc[brick_idx]
        brick_id = int(row["brick_id"])
        brick_close = row["date"]
        brick_open = float(row["open"])
        brick_size = float(row["brick_size"])
        is_uptrend = bool(row["uptrend"])
        duration_s = float(row["duration_seconds"]) if pd.notna(row["duration_seconds"]) else 0.0

        # ── Load ticks for this brick ───────────────────────────
        if prev_brick_close is None:
            # First brick — no ticks before it
            tick_vectors = np.zeros((0, 9), dtype=np.float32)
            metadata_rows.append({
                "brick_id": brick_id, "n_ticks": 0,
                "start_time": None, "end_time": None,
            })
        else:
            ticks = _load_ticks_range(prev_brick_close, brick_close, _tick_cache, TICK_DIR)

            if len(ticks) < 2:
                tick_vectors = np.zeros((0, 9), dtype=np.float32)
                metadata_rows.append({
                    "brick_id": brick_id, "n_ticks": 0,
                    "start_time": None, "end_time": None,
                })
            else:
                # Extract numpy arrays for speed
                bids = ticks["bid"].values
                asks = ticks["ask"].values
                timestamps_ms = ticks["timestamp"].values.view("int64") / 1e6  # ns → ms

                n_ticks = len(bids)
                vectors = np.zeros((n_ticks - 1, 9), dtype=np.float32)

                # Previous brick info for Flag_Zone
                if brick_idx > 0:
                    prev_row = labels.iloc[brick_idx - 1]
                    prev_brick_open = float(prev_row["open"])
                    prev_brick_size = float(prev_row["brick_size"])
                else:
                    prev_brick_open = brick_open
                    prev_brick_size = brick_size

                for i in range(1, n_ticks):
                    # ── VOLUME FALLBACK PROXY ──────────────────────
                    # Exactly as BrickOfTicks_Trader/bridge/feature_engine.py
                    mid_k = (bids[i] + asks[i]) / 2.0
                    mid_km1 = (bids[i-1] + asks[i-1]) / 2.0

                    if mid_k > mid_km1:
                        raw_ofi = 1.0
                    elif mid_k < mid_km1:
                        raw_ofi = -1.0
                    else:
                        raw_ofi = 0.0

                    depth_raw = 0.0
                    susc_raw = 0.0
                    # ── END FALLBACK ──────────────────────────────

                    vel_raw = compute_velocity(timestamps_ms[i], timestamps_ms[i-1])
                    spread_raw = compute_spread(asks[i], bids[i])

                    # Z-score (sequential, maintains state)
                    z_ofi = zs_ofi.update(raw_ofi)
                    z_depth = zs_depth.update(depth_raw)
                    z_susc = zs_susc.update(susc_raw)
                    z_vel = zs_vel.update(vel_raw)
                    z_spread = zs_spread.update(spread_raw)

                    # Non-z-scored features
                    mid = mid_k
                    progress = compute_progress(mid, brick_open, brick_size)
                    flag_curr = 1.0
                    flag_zone = float(compute_flag_zone(mid, prev_brick_open, prev_brick_size))
                    decay = 0.0

                    # Store
                    vec = [z_ofi, z_depth, z_susc, z_vel, z_spread,
                           progress, flag_curr, flag_zone, decay]
                    vectors[i - 1] = vec

                    # Stats tracking
                    for fi, v in enumerate(vec):
                        if not np.isfinite(v):
                            nan_inf_count += 1
                        else:
                            feat_stats[fi]["sum"] += v
                            feat_stats[fi]["sum2"] += v * v
                            feat_stats[fi]["n"] += 1
                            feat_stats[fi]["min"] = min(feat_stats[fi]["min"], v)
                            feat_stats[fi]["max"] = max(feat_stats[fi]["max"], v)

                tick_vectors = vectors
                total_ticks_processed += len(vectors)
                metadata_rows.append({
                    "brick_id": brick_id,
                    "n_ticks": len(vectors),
                    "start_time": ticks["timestamp"].iloc[1],
                    "end_time": ticks["timestamp"].iloc[-1],
                })

        # ── Save tick vectors ───────────────────────────────────
        np.save(FEATURE_DIR / f"tick_vectors_{brick_id}.npy", tick_vectors)

        # ── Compute macro-vector ────────────────────────────────
        macro = compute_macro_vector(duration_s, is_uptrend, brick_size, brick_size_history)
        macro_vectors[brick_idx] = macro
        brick_size_history.append(brick_size)

        # ── Update state ────────────────────────────────────────
        prev_brick_close = brick_close

        # ── Memory management ───────────────────────────────────
        if len(_tick_cache) > MAX_CACHE_SIZE:
            oldest = sorted(_tick_cache.keys())[:len(_tick_cache) - 3]
            for k in oldest:
                del _tick_cache[k]

        # ── Progress ────────────────────────────────────────────
        if (brick_idx + 1) % 1000 == 0 or brick_idx == n_bricks - 1:
            elapsed = time.time() - t_start
            rate = (brick_idx + 1) / elapsed
            eta = (n_bricks - brick_idx - 1) / rate if rate > 0 else 0
            print(f"  [{brick_idx+1:>6,}/{n_bricks:,}] ({(brick_idx+1)/n_bricks*100:.1f}%) "
                  f"ticks={total_ticks_processed:,}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    # ── Save macro vectors ──────────────────────────────────────
    np.save(FEATURE_DIR / "macro_vectors.npy", macro_vectors)

    # ── Save metadata ───────────────────────────────────────────
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_parquet(FEATURE_DIR / "brick_metadata.parquet", index=False)

    elapsed_total = time.time() - t_start
    print(f"\n✅ Done in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"   Total ticks processed: {total_ticks_processed:,}")
    print(f"   NaN/Inf count: {nan_inf_count}")

    return feat_stats, nan_inf_count


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════

def validate_features(feat_stats: dict, nan_inf_count: int, n_bricks: int) -> bool:
    """Validate fallback feature engineering output."""
    print("\n" + "=" * 60)
    print(" Fallback Feature Validation")
    print("=" * 60)

    all_pass = True

    # Check 1: No NaN/Inf
    check1 = nan_inf_count == 0
    print(f"\n✓ No NaN/Inf in features: {'PASS ✅' if check1 else f'FAIL ❌ ({nan_inf_count} found)'}")
    all_pass &= check1

    # Check 2: Feature statistics
    feature_names = ["z_OFI", "z_Depth", "z_Susc", "z_Vel", "z_Spread",
                     "Progress", "Flag_Curr", "Flag_Zone", "Decay"]
    print(f"\n📊 Feature statistics:")
    print(f"  {'Feature':12s}  {'Mean':>10s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}  {'N':>12s}")
    for i, name in enumerate(feature_names):
        s = feat_stats[i]
        n = s["n"]
        if n > 0:
            mean = s["sum"] / n
            var = (s["sum2"] / n) - mean ** 2
            std = sqrt(max(var, 0))
            print(f"  {name:12s}  {mean:10.4f}  {std:10.4f}  {s['min']:10.4f}  {s['max']:10.4f}  {n:12,}")
        else:
            print(f"  {name:12s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}  {0:12,}")

    # Check 3: z_Depth and z_Susc should be exactly 0.0 (constant input → z=0)
    depth_stats = feat_stats[1]
    susc_stats = feat_stats[2]
    if depth_stats["n"] > 0:
        depth_max_abs = max(abs(depth_stats["min"]), abs(depth_stats["max"]))
        check3 = depth_max_abs < 1e-10
        print(f"\n✓ z_Depth ≡ 0.0 (max |z_Depth| = {depth_max_abs:.2e}): "
              f"{'PASS ✅' if check3 else 'FAIL ❌'}")
        all_pass &= check3

    if susc_stats["n"] > 0:
        susc_max_abs = max(abs(susc_stats["min"]), abs(susc_stats["max"]))
        check3b = susc_max_abs < 1e-10
        print(f"✓ z_Susc ≡ 0.0 (max |z_Susc| = {susc_max_abs:.2e}): "
              f"{'PASS ✅' if check3b else 'FAIL ❌'}")
        all_pass &= check3b

    # Check 4: z_OFI should only have values derived from {-1, 0, +1} inputs
    ofi_stats = feat_stats[0]
    if ofi_stats["n"] > 0:
        print(f"✓ z_OFI range: [{ofi_stats['min']:.4f}, {ofi_stats['max']:.4f}] "
              f"(from tick-direction proxy)")

    # Check 5: Macro vectors saved
    macro_path = FEATURE_DIR / "macro_vectors.npy"
    if macro_path.exists():
        mv = np.load(macro_path)
        check5 = mv.shape == (n_bricks, 3)
        print(f"\n✓ Macro vectors shape {mv.shape}: {'PASS ✅' if check5 else 'FAIL ❌'}")
        all_pass &= check5

    # Check 6: Tick vector files count
    tick_files = list(FEATURE_DIR.glob("tick_vectors_*.npy"))
    check6 = len(tick_files) == n_bricks
    print(f"✓ Tick vector files: {len(tick_files)}/{n_bricks} "
          f"{'PASS ✅' if check6 else 'FAIL ❌'}")
    all_pass &= check6

    print("\n" + "=" * 60)
    print(f" {'ALL CHECKS PASSED ✅' if all_pass else 'SOME CHECKS FAILED ❌'}")
    print("=" * 60)

    return all_pass


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Volume Fallback: Feature Engineering")
    print("=" * 60)

    if not LABELS_PATH.exists():
        print(f"ERROR: {LABELS_PATH} not found.")
        sys.exit(1)

    labels = pd.read_parquet(LABELS_PATH)
    n_bricks = len(labels)

    feat_stats, nan_inf_count = process_all_ticks_fallback()
    passed = validate_features(feat_stats, nan_inf_count, n_bricks)

    if not passed:
        print("\n⚠️  Some validation checks failed. Review output above.")
        sys.exit(1)

    print(f"\n💾 Fallback features saved to {FEATURE_DIR}")


if __name__ == "__main__":
    main()
