"""
Phase 1: Label Generation — Hybrid Overshoot Algorithm

Enriches every Renko brick with y_class, y_mag, duration_seconds by scanning L1 tick data.

Key conventions:
- entry_price = brick.close (bid-based, preserves per-brick ordering)
- TP/SL levels computed from entry_price ± brick_size
- ALL tick scanning uses mid-price: mid = (bid + ask) / 2
- y_class derived from hybrid algorithm (tp_hit), NOT from CSV outcome column
- CSV outcome retained as validation reference only
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import warnings

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "Raw"
TICK_DIR = DATA_DIR / "Ticks"
OUTPUT_DIR = BASE_DIR / "outputs"

# Maximum number of trading days to scan forward for overshoot resolution
MAX_SCAN_DAYS = 5


# ═══════════════════════════════════════════════════════════════
# 1.1 Core Functions
# ═══════════════════════════════════════════════════════════════

def load_ticks_for_date(date: pd.Timestamp) -> pd.DataFrame:
    """Load the tick parquet file for a given trading date.

    Args:
        date: A pandas Timestamp. The year/month/day are used to locate
              the file at Data/Raw/Ticks/{year}/{month:02d}/{day:02d}.parquet

    Returns:
        DataFrame with columns [timestamp, bid, bid_vol, ask, ask_vol]
        or empty DataFrame if file not found.
    """
    path = TICK_DIR / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "bid", "bid_vol", "ask", "ask_vol"])
    df = pd.read_parquet(path)
    # Ensure timestamp is tz-aware UTC
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def load_ticks_range(start_time: pd.Timestamp, max_days: int = MAX_SCAN_DAYS) -> pd.DataFrame:
    """Load and concatenate tick files from start_time forward up to max_days trading days.

    Scans sequentially through calendar days, skipping missing files (weekends/holidays).
    Stops after accumulating max_days worth of actual trading-day files.

    Args:
        start_time: UTC timestamp to start loading ticks from.
        max_days: Maximum number of trading-day files to load.

    Returns:
        DataFrame of ticks from start_time onwards, sorted by timestamp.
    """
    # Ensure UTC
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")

    frames = []
    trading_days_loaded = 0
    current_date = start_time.normalize()  # Start of this calendar day

    # Scan forward through calendar days
    max_calendar_days = max_days * 3  # Account for weekends/holidays
    for offset in range(max_calendar_days):
        check_date = current_date + timedelta(days=offset)
        ticks = load_ticks_for_date(check_date)
        if len(ticks) > 0:
            frames.append(ticks)
            trading_days_loaded += 1
            if trading_days_loaded >= max_days:
                break

    if not frames:
        return pd.DataFrame(columns=["timestamp", "bid", "bid_vol", "ask", "ask_vol"])

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)

    # Filter to ticks at or after start_time
    combined = combined[combined["timestamp"] >= start_time].reset_index(drop=True)
    return combined


def calculate_true_overshoot(entry: float, brick_size: float, is_long: bool,
                              future_ticks: pd.DataFrame) -> dict:
    """Compute y_mag using the hybrid overshoot algorithm (FR-LG-01).

    Phase 1 (pre-TP): Scan ticks using mid-price. Track peak extension.
        Reversal = fixed SL level. If SL hit → LOSS.
    Phase 2 (post-TP): Switch to dynamic 1-brick-size trailing reversal from peak.
        Continue until trailing reversal triggered.

    Args:
        entry: Entry price (brick.close, bid-based).
        brick_size: Brick size in price units.
        is_long: True for LONG (uptrend), False for SHORT.
        future_ticks: DataFrame with columns [timestamp, bid, ask].
                      Must contain ticks AFTER the brick close.

    Returns:
        dict with keys:
            y_mag (float): abs(peak - entry) / brick_size
            y_class (int): 1 if TP hit, 0 otherwise
            tp_hit (bool): Whether TP was reached
            resolved (bool): Whether scan reached a conclusion (SL/trailing reversal)
    """
    if len(future_ticks) == 0:
        return {"y_mag": None, "y_class": None, "tp_hit": False, "resolved": False}

    # TP and SL levels
    tp = entry + brick_size if is_long else entry - brick_size
    sl = entry - brick_size if is_long else entry + brick_size

    peak = entry  # Rolling max (LONG) or min (SHORT)
    tp_hit = False

    # Extract arrays for speed
    bids = future_ticks["bid"].values
    asks = future_ticks["ask"].values

    for i in range(len(bids)):
        mid = (bids[i] + asks[i]) / 2.0

        # Track peak extension in favorable direction
        if is_long:
            peak = max(peak, mid)
        else:
            peak = min(peak, mid)

        # Phase 1: Before TP — check SL (fixed level)
        if not tp_hit:
            if (is_long and mid >= tp) or (not is_long and mid <= tp):
                tp_hit = True
                continue  # TP just hit, don't check reversal on same tick
            if (is_long and mid <= sl) or (not is_long and mid >= sl):
                break  # SL hit → LOSS
        # Phase 2: After TP — trailing reversal check
        else:
            if is_long and mid <= peak - brick_size:
                break
            if not is_long and mid >= peak + brick_size:
                break
    else:
        # Loop completed without break → tick data exhausted before resolution
        if not tp_hit:
            # Never hit TP or SL — unresolved
            y_mag = abs(peak - entry) / brick_size
            return {"y_mag": y_mag, "y_class": None, "tp_hit": False, "resolved": False}

    y_mag = abs(peak - entry) / brick_size
    y_class = 1 if tp_hit else 0

    return {"y_mag": y_mag, "y_class": y_class, "tp_hit": tp_hit, "resolved": True}


def compute_duration(brick_close_time: pd.Timestamp,
                     next_brick_close_time: pd.Timestamp) -> float:
    """Compute duration in seconds between brick close and next brick close.

    Args:
        brick_close_time: Timestamp of this brick's close.
        next_brick_close_time: Timestamp of the next brick's close.

    Returns:
        Duration in seconds (float). Returns NaN for the last brick.
    """
    if pd.isna(next_brick_close_time):
        return np.nan
    return (next_brick_close_time - brick_close_time).total_seconds()


# ═══════════════════════════════════════════════════════════════
# 1.2 Batch Processing
# ═══════════════════════════════════════════════════════════════

def generate_all_labels(renko_csv_path: str = None, tick_dir: str = None) -> pd.DataFrame:
    """Generate y_class, y_mag, duration_seconds for every brick in the Renko CSV.

    Args:
        renko_csv_path: Path to Renko CSV (defaults to standard location).
        tick_dir: Path to tick data directory (defaults to standard location).

    Returns:
        DataFrame with all original columns plus:
        brick_id, y_class, y_mag, duration_seconds, exclude_flag, csv_outcome_match
    """
    global TICK_DIR

    if renko_csv_path is None:
        renko_csv_path = DATA_DIR / "renko_with_tick_outcomes_no_be_XAUUSD20-24.csv"
    if tick_dir is not None:
        TICK_DIR = Path(tick_dir)

    # ── Load Renko CSV ─────────────────────────────────────────
    df = pd.read_csv(renko_csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    n = len(df)
    print(f"Loaded {n:,} bricks from {Path(renko_csv_path).name}")

    # ── Initialize output columns ──────────────────────────────
    df["brick_id"] = range(n)
    df["y_class"] = np.nan
    df["y_mag"] = np.nan
    df["duration_seconds"] = np.nan
    df["exclude_flag"] = False
    df["csv_outcome_match"] = np.nan

    # ── Compute duration_seconds ───────────────────────────────
    df["duration_seconds"] = df["date"].shift(-1).sub(df["date"]).dt.total_seconds()
    # Last brick has NaN duration (no next brick)

    # ── Tick data cache (avoid reloading same file) ────────────
    _tick_cache = {}

    def _get_cached_ticks(date_key: pd.Timestamp) -> pd.DataFrame:
        """Load tick file with caching by date."""
        key = (date_key.year, date_key.month, date_key.day)
        if key not in _tick_cache:
            _tick_cache[key] = load_ticks_for_date(date_key)
        return _tick_cache[key]

    # ── Process each brick ─────────────────────────────────────
    excluded_count = 0
    resolved_count = 0

    for idx in range(n):
        row = df.iloc[idx]
        brick_close_time = row["date"]
        entry_price = row["close"]      # bid-based entry
        brick_size = row["brick_size"]
        is_long = bool(row["uptrend"])

        # Load ticks starting from brick close time
        # First try the current trading day's file
        current_date = brick_close_time.normalize()
        ticks = _get_cached_ticks(current_date)

        if len(ticks) > 0:
            # Filter to ticks AFTER brick close
            future_mask = ticks["timestamp"] > brick_close_time
            future_ticks = ticks[future_mask]

            # If very few ticks left in this day's file, load next day too
            if len(future_ticks) < 100:
                next_ticks = _load_additional_days(current_date, brick_close_time,
                                                   _tick_cache, max_extra_days=MAX_SCAN_DAYS - 1)
                if len(next_ticks) > 0:
                    future_ticks = pd.concat([future_ticks, next_ticks], ignore_index=True)
        else:
            # No tick file for this date, try loading range
            future_ticks = load_ticks_range(brick_close_time, max_days=MAX_SCAN_DAYS)

        # Run hybrid overshoot algorithm
        result = calculate_true_overshoot(entry_price, brick_size, is_long, future_ticks)

        if not result["resolved"] or result["y_class"] is None:
            df.at[idx, "exclude_flag"] = True
            excluded_count += 1
        else:
            df.at[idx, "y_class"] = result["y_class"]
            df.at[idx, "y_mag"] = result["y_mag"]
            resolved_count += 1

            # CSV outcome comparison
            csv_outcome_win = 1 if row["outcome"] == "WIN" else 0
            df.at[idx, "csv_outcome_match"] = float(result["y_class"] == csv_outcome_win)

        # Progress reporting
        if (idx + 1) % 1000 == 0 or idx == n - 1:
            pct = (idx + 1) / n * 100
            print(f"  [{idx+1:>6,}/{n:,}] ({pct:5.1f}%)  resolved={resolved_count:,}  "
                  f"excluded={excluded_count:,}")

        # Evict old cache entries to manage memory (keep only last 3 dates)
        if len(_tick_cache) > 5:
            oldest_keys = sorted(_tick_cache.keys())[:len(_tick_cache) - 3]
            for k in oldest_keys:
                del _tick_cache[k]

    # ── Cast types ─────────────────────────────────────────────
    resolved_mask = ~df["exclude_flag"]
    df.loc[resolved_mask, "y_class"] = df.loc[resolved_mask, "y_class"].astype(int)

    print(f"\nDone: {resolved_count:,} resolved, {excluded_count:,} excluded")
    return df


def _load_additional_days(current_date: pd.Timestamp, start_time: pd.Timestamp,
                           cache: dict, max_extra_days: int = 4) -> pd.DataFrame:
    """Load tick data from subsequent trading days beyond the current file.

    Args:
        current_date: The current trading date (normalized).
        start_time: The brick close timestamp (filter ticks after this).
        cache: The tick cache dict to use/populate.
        max_extra_days: Max additional trading day files to load.

    Returns:
        DataFrame of ticks from next trading days, after start_time.
    """
    frames = []
    days_loaded = 0
    max_calendar = max_extra_days * 3  # Account for weekends

    for offset in range(1, max_calendar + 1):
        check_date = current_date + timedelta(days=offset)
        key = (check_date.year, check_date.month, check_date.day)

        if key not in cache:
            cache[key] = load_ticks_for_date(check_date)

        ticks = cache[key]
        if len(ticks) > 0:
            # All ticks from next day are "future" relative to current brick
            future = ticks[ticks["timestamp"] > start_time]
            if len(future) > 0:
                frames.append(future)
            days_loaded += 1
            if days_loaded >= max_extra_days:
                break

    if not frames:
        return pd.DataFrame(columns=["timestamp", "bid", "bid_vol", "ask", "ask_vol"])
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# 1.3 Validation
# ═══════════════════════════════════════════════════════════════

def validate_labels(df: pd.DataFrame) -> bool:
    """Run all Phase 1 validation checks on the generated labels.

    Returns True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print(" Phase 1 Validation")
    print("=" * 60)

    resolved = df[~df["exclude_flag"]].copy()
    n_resolved = len(resolved)
    n_excluded = df["exclude_flag"].sum()
    all_pass = True

    # ── Check 1: y_mag >= 0 for all resolved ──────────────────
    check1 = (resolved["y_mag"] >= 0.0).all()
    print(f"\n✓ y_mag >= 0.0 for all resolved: {'PASS ✅' if check1 else 'FAIL ❌'}")
    all_pass &= check1

    # ── Check 2: y_mag bounds by y_class ──────────────────────
    loss_mask = resolved["y_class"] == 0
    win_mask = resolved["y_class"] == 1

    check2a = (resolved.loc[loss_mask, "y_mag"] < 1.0).all()
    check2b = (resolved.loc[win_mask, "y_mag"] >= 1.0).all()
    print(f"✓ y_mag < 1.0 for LOSS (y_class=0): {'PASS ✅' if check2a else 'FAIL ❌'}")
    print(f"✓ y_mag >= 1.0 for WIN (y_class=1):  {'PASS ✅' if check2b else 'FAIL ❌'}")
    all_pass &= check2a
    all_pass &= check2b

    # ── Check 3: CSV outcome mismatch rate ────────────────────
    valid_match = resolved["csv_outcome_match"].dropna()
    n_matched = int(valid_match.sum())
    n_mismatched = len(valid_match) - n_matched
    mismatch_rate = n_mismatched / len(valid_match) * 100 if len(valid_match) > 0 else 0

    check3 = mismatch_rate <= 15.0
    print(f"\n📊 CSV outcome mismatch rate: {mismatch_rate:.1f}% "
          f"({n_mismatched:,}/{len(valid_match):,})")
    print(f"   Expected 5-10%. {'PASS ✅' if check3 else 'INVESTIGATE ⚠️'}")
    all_pass &= check3

    if not check3:
        # Print y_mag of mismatched bricks for investigation
        mismatched = resolved[resolved["csv_outcome_match"] == 0.0]
        print(f"   Mismatched bricks y_mag stats:")
        print(f"   mean={mismatched['y_mag'].mean():.3f}, "
              f"median={mismatched['y_mag'].median():.3f}")

    # ── Check 4: Mismatch directional consistency ──────────────
    # The systematic mid-vs-bid effect means:
    #   LONG mismatches should be mostly CSV=LOSS→algo=WIN (mid reaches TP sooner)
    #   SHORT mismatches should be mostly CSV=WIN→algo=LOSS (mid reaches TP later)
    mismatched = resolved[resolved["csv_outcome_match"] == 0.0]
    if len(mismatched) > 0:
        long_mm = mismatched[mismatched["uptrend"] == True]
        short_mm = mismatched[mismatched["uptrend"] == False]

        long_correct_dir = long_mm[
            (long_mm["outcome"] == "LOSS") & (long_mm["y_class"] == 1)
        ] if len(long_mm) > 0 else pd.DataFrame()
        short_correct_dir = short_mm[
            (short_mm["outcome"] == "WIN") & (short_mm["y_class"] == 0)
        ] if len(short_mm) > 0 else pd.DataFrame()

        total_directional = len(long_correct_dir) + len(short_correct_dir)
        dir_pct = total_directional / len(mismatched) * 100
        check4 = dir_pct >= 90.0

        print(f"\n📊 Mismatch directional consistency: {dir_pct:.1f}%")
        print(f"   LONG mismatches: {len(long_mm):,} "
              f"(LOSS→WIN: {len(long_correct_dir):,}, other: {len(long_mm)-len(long_correct_dir):,})")
        print(f"   SHORT mismatches: {len(short_mm):,} "
              f"(WIN→LOSS: {len(short_correct_dir):,}, other: {len(short_mm)-len(short_correct_dir):,})")
        print(f"   Expected >90% directional. {'PASS ✅' if check4 else 'INVESTIGATE ⚠️'}")
        all_pass &= check4

        # Also report boundary clustering as informational
        in_boundary = ((mismatched["y_mag"] >= 0.85) & (mismatched["y_mag"] <= 1.15)).sum()
        boundary_pct = in_boundary / len(mismatched) * 100
        print(f"   Boundary [0.85,1.15] clustering: {boundary_pct:.1f}% (informational)")
    else:
        print(f"\n📊 No mismatches to analyze")

    # ── Check 5: y_mag distribution stats ─────────────────────
    print(f"\n📈 y_mag distribution by class:")
    for cls, label in [(0, "LOSS"), (1, "WIN")]:
        subset = resolved[resolved["y_class"] == cls]["y_mag"]
        if len(subset) > 0:
            print(f"   {label} (n={len(subset):,}): "
                  f"mean={subset.mean():.3f}, median={subset.median():.3f}, "
                  f"std={subset.std():.3f}, min={subset.min():.3f}, max={subset.max():.3f}")

    # ── Check 6: Excluded bricks count ────────────────────────
    print(f"\n📊 Excluded bricks (tick data gaps): {n_excluded:,} "
          f"({n_excluded/len(df)*100:.2f}%)")
    print(f"   Resolved bricks: {n_resolved:,} ({n_resolved/len(df)*100:.2f}%)")

    # ── Overall WIN/LOSS ratio ────────────────────────────────
    n_win = (resolved["y_class"] == 1).sum()
    n_loss = (resolved["y_class"] == 0).sum()
    print(f"\n📊 y_class ratio: WIN={n_win:,} ({n_win/n_resolved*100:.1f}%), "
          f"LOSS={n_loss:,} ({n_loss/n_resolved*100:.1f}%)")

    # ── Duration stats ────────────────────────────────────────
    dur = resolved["duration_seconds"].dropna()
    if len(dur) > 0:
        print(f"\n⏱  Duration stats (seconds):")
        print(f"   mean={dur.mean():.1f}, median={dur.median():.1f}, "
              f"std={dur.std():.1f}")
        print(f"   min={dur.min():.1f}, max={dur.max():.1f}")
        fast_bricks = (dur < 2.0).sum()
        print(f"   Bricks with duration < 2s: {fast_bricks:,} "
              f"({fast_bricks/len(dur)*100:.1f}%)")

    print("\n" + "=" * 60)
    status = "ALL CHECKS PASSED ✅" if all_pass else "SOME CHECKS FAILED ❌"
    print(f" {status}")
    print("=" * 60)

    return all_pass


# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    """Run the full Phase 1 pipeline: generate labels → validate → save."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate labels
    df = generate_all_labels()

    # Validate
    passed = validate_labels(df)

    # Save to parquet (fix dtypes for pyarrow compatibility)
    save_df = df.copy()
    # sequence column contains large integers stored as Python objects — cast to string
    if "sequence" in save_df.columns:
        save_df["sequence"] = save_df["sequence"].astype(str)
    out_path = OUTPUT_DIR / "labels.parquet"
    save_df.to_parquet(out_path, index=False)
    print(f"\n💾 Saved to {out_path} ({len(df):,} rows)")

    if not passed:
        print("\n⚠️  Some validation checks failed. Review output above.")

    return df


if __name__ == "__main__":
    main()
