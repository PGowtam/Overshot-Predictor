"""
Phase 2: Feature Engineering (FR-FE-01 through FR-FE-07)

Computes the 9D tick feature vector and 3D macro-vector for every brick.

9D Tick Vector:
  [0] z_OFI      — z-scored Order Flow Imbalance (weak inequalities)
  [1] z_Depth    — z-scored total depth (bid_vol + ask_vol)
  [2] z_Susc     — z-scored susceptibility (raw OFI / raw Depth)
  [3] z_Vel      — z-scored tick velocity (1 / dt_ms)
  [4] z_Spread   — z-scored spread (ask - bid)
  [5] Progress   — (mid - brick_open) / brick_size  (sawtooth, resets each brick)
  [6] Flag_Curr  — 1 if tick is in current brick, 0 otherwise
  [7] Flag_Zone  — 1 if mid is beyond previous brick boundary
  [8] Decay      — (current_brick_id - tick_brick_id) / max_depth

3D Macro Vector (per brick):
  [0] log_dur    — log(duration_seconds + 1)
  [1] direction  — +1 (uptrend) or -1 (downtrend)
  [2] z_size     — (brick_size - mean_50) / std_50
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
FEATURE_DIR = OUTPUT_DIR / "features"


# ═══════════════════════════════════════════════════════════════
# 2.1 Rolling Z-Score (FR-FE-03, FR-FE-04, FR-FE-05)
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
# 2.2–2.3 Raw Feature Computations
# ═══════════════════════════════════════════════════════════════

def compute_ofi(bid_k: float, bid_km1: float, ask_k: float, ask_km1: float,
                bid_vol_k: float, bid_vol_km1: float,
                ask_vol_k: float, ask_vol_km1: float) -> float:
    """Compute OFI using WEAK inequalities (FR-FE-01).

    e_k = I(dBid>=0)*q^B_k - I(dBid<=0)*q^B_{k-1}
        - I(dAsk<=0)*q^A_k + I(dAsk>=0)*q^A_{k-1}
    """
    dBid = bid_k - bid_km1
    dAsk = ask_k - ask_km1

    e_k = (
        (1 if dBid >= 0 else 0) * bid_vol_k
      - (1 if dBid <= 0 else 0) * bid_vol_km1
      - (1 if dAsk <= 0 else 0) * ask_vol_k
      + (1 if dAsk >= 0 else 0) * ask_vol_km1
    )
    return float(e_k)


def compute_depth(bid_vol: float, ask_vol: float) -> float:
    """Compute total depth."""
    return bid_vol + ask_vol


def compute_susceptibility(ofi_raw: float, depth_raw: float) -> float:
    """Compute susceptibility: divide RAW first, THEN z-score (FR-FE-02)."""
    return ofi_raw / (depth_raw + 1e-8)


def compute_velocity(t_k_ms: float, t_km1_ms: float) -> float:
    """Compute tick velocity: 1 / (dt_ms + 1e-3)."""
    dt = t_k_ms - t_km1_ms
    return 1.0 / (dt + 1e-3)


def compute_spread(ask: float, bid: float) -> float:
    """Compute spread."""
    return ask - bid


def compute_progress(mid: float, brick_open: float, brick_size: float) -> float:
    """Compute progress: (mid - brick_open) / brick_size. Sawtooth, resets each brick."""
    return (mid - brick_open) / brick_size


def compute_flag_curr(tick_brick_id: int, current_brick_id: int) -> int:
    """1 if tick belongs to current brick, 0 otherwise."""
    return 1 if tick_brick_id == current_brick_id else 0


def compute_flag_zone(mid: float, prev_brick_open: float, prev_brick_size: float) -> int:
    """1 if mid is beyond the previous brick's boundary (post-outcome zone)."""
    return 1 if abs(mid - prev_brick_open) >= prev_brick_size else 0


def compute_decay(current_brick_id: int, tick_brick_id: int, max_depth: int) -> float:
    """Decay: 0=current brick, 1=oldest in buffer. max_depth is buffer capacity."""
    if max_depth <= 0:
        return 0.0
    return min((current_brick_id - tick_brick_id) / max_depth, 1.0)


# ═══════════════════════════════════════════════════════════════
# 2.4 Macro-Vector Computation (FR-FE-07)
# ═══════════════════════════════════════════════════════════════

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
        recent = brick_size_history[-50:]  # last 50 or fewer
        mu = np.mean(recent)
        sigma = np.std(recent, ddof=1) if len(recent) > 1 else 0.0
        if sigma < 1e-12:
            z_size = 0.0
        else:
            z_size = (brick_size - mu) / sigma

    return np.array([log_dur, direction, z_size], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# 2.5 Full Pipeline
# ═══════════════════════════════════════════════════════════════

def load_ticks_for_date(date: pd.Timestamp) -> pd.DataFrame:
    """Load tick parquet for a given date."""
    path = TICK_DIR / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "bid", "bid_vol", "ask", "ask_vol"])
    df = pd.read_parquet(path)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def process_all_ticks(labels_path: str = None, tick_dir: str = None):
    """Process all ticks and compute 9D feature vectors + 3D macro vectors.

    Processes brick-by-brick: for each brick, loads ticks from previous
    brick close to current brick close. Z-score deques are maintained
    across bricks (never reset).

    Args:
        labels_path: Path to labels.parquet (default: outputs/labels.parquet).
        tick_dir: Path to tick directory (default: Data/Raw/Ticks).

    Outputs:
        - outputs/features/tick_vectors_{brick_id}.npy per brick
        - outputs/features/macro_vectors.npy
        - outputs/features/brick_metadata.parquet
    """
    # Setup paths
    if labels_path is None:
        labels_path = OUTPUT_DIR / "labels.parquet"
    if tick_dir is None:
        tick_dir = TICK_DIR

    # Load labels
    labels = pd.read_parquet(labels_path)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    n_bricks = len(labels)
    print(f"Loaded {n_bricks:,} bricks")

    # Create output dirs
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize rolling z-score instances (never reset across bricks)
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
        # Ticks between previous brick close and this brick close
        if prev_brick_close is None:
            # First brick — no ticks before it
            tick_vectors = np.zeros((0, 9), dtype=np.float32)
            metadata_rows.append({
                "brick_id": brick_id, "n_ticks": 0,
                "start_time": None, "end_time": None,
            })
        else:
            # Find ticks between prev close and current close
            ticks = _load_ticks_range(prev_brick_close, brick_close, _tick_cache, tick_dir)

            if len(ticks) < 2:
                # Not enough ticks for delta computation
                tick_vectors = np.zeros((0, 9), dtype=np.float32)
                metadata_rows.append({
                    "brick_id": brick_id, "n_ticks": 0,
                    "start_time": None, "end_time": None,
                })
            else:
                # Extract numpy arrays for speed
                bids = ticks["bid"].values
                asks = ticks["ask"].values
                bid_vols = ticks["bid_vol"].values
                ask_vols = ticks["ask_vol"].values
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
                    # Raw features
                    ofi_raw = compute_ofi(
                        bids[i], bids[i-1], asks[i], asks[i-1],
                        bid_vols[i], bid_vols[i-1], ask_vols[i], ask_vols[i-1]
                    )
                    depth_raw = compute_depth(bid_vols[i], ask_vols[i])
                    susc_raw = compute_susceptibility(ofi_raw, depth_raw)
                    vel_raw = compute_velocity(timestamps_ms[i], timestamps_ms[i-1])
                    spread_raw = compute_spread(asks[i], bids[i])

                    # Z-score (sequential, maintains state)
                    z_ofi = zs_ofi.update(ofi_raw)
                    z_depth = zs_depth.update(depth_raw)
                    z_susc = zs_susc.update(susc_raw)
                    z_vel = zs_vel.update(vel_raw)
                    z_spread = zs_spread.update(spread_raw)

                    # Non-z-scored features
                    mid = (bids[i] + asks[i]) / 2.0
                    progress = compute_progress(mid, brick_open, brick_size)
                    flag_curr = 1.0  # All ticks in range are "current" brick
                    flag_zone = float(compute_flag_zone(mid, prev_brick_open, prev_brick_size))
                    decay = 0.0  # Current brick, oldest ticks get decay in buffer phase

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


def _load_ticks_range(start_time: pd.Timestamp, end_time: pd.Timestamp,
                      cache: dict, tick_dir: Path) -> pd.DataFrame:
    """Load ticks between start_time and end_time, spanning multiple days if needed."""
    # Generate date range
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
# 2.6 Validation
# ═══════════════════════════════════════════════════════════════

def validate_features(feat_stats: dict, nan_inf_count: int, n_bricks: int) -> bool:
    """Validate feature engineering output."""
    print("\n" + "=" * 60)
    print(" Phase 2 Validation")
    print("=" * 60)

    all_pass = True

    # Check 1: No NaN/Inf in any feature (especially z_Susc)
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

    # Check 3: Macro vectors saved
    macro_path = FEATURE_DIR / "macro_vectors.npy"
    if macro_path.exists():
        mv = np.load(macro_path)
        check3 = mv.shape == (n_bricks, 3)
        print(f"\n✓ Macro vectors shape {mv.shape}: {'PASS ✅' if check3 else 'FAIL ❌'}")
        all_pass &= check3
    else:
        print(f"\n✓ Macro vectors: FAIL ❌ (file not found)")
        all_pass = False

    # Check 4: Tick vector files count
    tick_files = list(FEATURE_DIR.glob("tick_vectors_*.npy"))
    check4 = len(tick_files) == n_bricks
    print(f"✓ Tick vector files: {len(tick_files)}/{n_bricks} "
          f"{'PASS ✅' if check4 else 'FAIL ❌'}")
    all_pass &= check4

    # Check 5: Spot-check progress reset at brick boundaries
    print(f"\n📊 Progress spot-check (first 5 bricks with ticks):")
    checked = 0
    for bid in range(n_bricks):
        path = FEATURE_DIR / f"tick_vectors_{bid}.npy"
        if path.exists():
            vecs = np.load(path)
            if len(vecs) > 0:
                prog_first = vecs[0, 5]  # Progress column
                prog_last = vecs[-1, 5]
                print(f"  brick_id={bid}: {len(vecs)} ticks, "
                      f"progress[0]={prog_first:.4f}, progress[-1]={prog_last:.4f}")
                checked += 1
                if checked >= 5:
                    break

    print("\n" + "=" * 60)
    print(f" {'ALL CHECKS PASSED ✅' if all_pass else 'SOME CHECKS FAILED ❌'}")
    print("=" * 60)

    return all_pass


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Phase 2: Feature Engineering")
    print("=" * 60)

    labels_path = OUTPUT_DIR / "labels.parquet"
    if not labels_path.exists():
        print("ERROR: labels.parquet not found. Run Phase 1 first.")
        sys.exit(1)

    labels = pd.read_parquet(labels_path)
    n_bricks = len(labels)

    feat_stats, nan_inf_count = process_all_ticks()
    passed = validate_features(feat_stats, nan_inf_count, n_bricks)

    if not passed:
        print("\n⚠️  Some validation checks failed. Review output above.")
        sys.exit(1)

    print(f"\n💾 Features saved to {FEATURE_DIR}")


if __name__ == "__main__":
    main()
