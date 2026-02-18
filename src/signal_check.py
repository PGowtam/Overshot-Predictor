"""
Phase 1.5: Signal Existence Checkpoint 1 (FR-SC-01)

Verifies that raw microstructure features carry measurable signal for
predicting y_class BEFORE investing in expensive feature engineering.

Extracts 3 raw features at each brick's close tick:
  - raw_ofi: Order Flow Imbalance (e_k) using weak inequalities
  - raw_velocity: tick arrival rate 1 / (dt + 1e-3)
  - raw_spread: ask - bid

Computes point-biserial correlation with y_class and applies decision gate:
  - RED (all |r| < 0.02): No linear signal
  - GREEN (any |r| > 0.03): Signal confirmed
  - AMBER (0.02–0.03): Weak signal, rely on non-linear patterns
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import json
import sys

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "Raw"
TICK_DIR = DATA_DIR / "Ticks"
OUTPUT_DIR = BASE_DIR / "outputs"


# ═══════════════════════════════════════════════════════════════
# 1.5.1 Raw Feature Extraction
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


def compute_raw_ofi(tick_k: pd.Series, tick_km1: pd.Series) -> float:
    """Compute raw OFI (e_k) using weak inequalities (FR-FE-01).

    e_k = I(dBid>=0)*q^B_k - I(dBid<=0)*q^B_{k-1}
        - I(dAsk<=0)*q^A_k + I(dAsk>=0)*q^A_{k-1}

    Args:
        tick_k: Current tick (bid, bid_vol, ask, ask_vol).
        tick_km1: Previous tick.

    Returns:
        Raw OFI value (float).
    """
    dBid = tick_k["bid"] - tick_km1["bid"]
    dAsk = tick_k["ask"] - tick_km1["ask"]

    e_k = (
        (1 if dBid >= 0 else 0) * tick_k["bid_vol"]
      - (1 if dBid <= 0 else 0) * tick_km1["bid_vol"]
      - (1 if dAsk <= 0 else 0) * tick_k["ask_vol"]
      + (1 if dAsk >= 0 else 0) * tick_km1["ask_vol"]
    )
    return float(e_k)


def compute_raw_velocity(tick_k: pd.Series, tick_km1: pd.Series) -> float:
    """Compute raw tick velocity: 1 / (dt_ms + 1e-3).

    dt is in milliseconds for numeric stability.
    """
    dt_ms = (tick_k["timestamp"] - tick_km1["timestamp"]).total_seconds() * 1000
    return 1.0 / (dt_ms + 1e-3)


def compute_raw_spread(tick_k: pd.Series) -> float:
    """Compute raw spread: ask - bid."""
    return float(tick_k["ask"] - tick_k["bid"])


def extract_raw_features(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Extract raw features at each brick's close tick.

    For each non-excluded brick:
      1. Load tick data for the brick's close date
      2. Find the last tick at or before brick close time
      3. Compute raw_ofi, raw_velocity, raw_spread

    Args:
        labels_df: DataFrame from outputs/labels.parquet.

    Returns:
        DataFrame with columns: brick_id, raw_ofi, raw_velocity, raw_spread, y_class
    """
    resolved = labels_df[~labels_df["exclude_flag"]].copy()
    n = len(resolved)

    results = []
    _tick_cache = {}
    skipped = 0

    for i, (idx, row) in enumerate(resolved.iterrows()):
        brick_close = row["date"]
        current_date = brick_close.normalize()

        # Load ticks (with cache)
        key = (current_date.year, current_date.month, current_date.day)
        if key not in _tick_cache:
            _tick_cache[key] = load_ticks_for_date(current_date)

        ticks = _tick_cache[key]

        if len(ticks) < 2:
            skipped += 1
            continue

        # Find last tick at or before brick close
        mask = ticks["timestamp"] <= brick_close
        before_ticks = ticks[mask]

        if len(before_ticks) < 2:
            skipped += 1
            continue

        tick_k = before_ticks.iloc[-1]
        tick_km1 = before_ticks.iloc[-2]

        raw_ofi = compute_raw_ofi(tick_k, tick_km1)
        raw_vel = compute_raw_velocity(tick_k, tick_km1)
        raw_spread = compute_raw_spread(tick_k)

        results.append({
            "brick_id": row["brick_id"],
            "raw_ofi": raw_ofi,
            "raw_velocity": raw_vel,
            "raw_spread": raw_spread,
            "y_class": int(row["y_class"]),
        })

        # Progress
        if (i + 1) % 5000 == 0 or i == n - 1:
            print(f"  [{i+1:>6,}/{n:,}] extracted, skipped={skipped}")

        # Memory management
        if len(_tick_cache) > 5:
            oldest = sorted(_tick_cache.keys())[:len(_tick_cache) - 3]
            for k in oldest:
                del _tick_cache[k]

    print(f"\nExtracted features for {len(results):,} bricks, skipped {skipped}")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 1.5.2 Correlation Analysis
# ═══════════════════════════════════════════════════════════════

def run_correlation_analysis(features_df: pd.DataFrame) -> dict:
    """Compute point-biserial correlations between raw features and y_class.

    Returns dict of {feature_name: {r, p_value, abs_r}}.
    """
    feature_names = ["raw_ofi", "raw_velocity", "raw_spread"]
    y = features_df["y_class"].values

    results = {}
    for feat in feature_names:
        x = features_df[feat].values

        # Remove NaN/Inf
        valid = np.isfinite(x)
        x_clean = x[valid]
        y_clean = y[valid]

        if len(x_clean) < 10:
            results[feat] = {"r": 0.0, "p_value": 1.0, "abs_r": 0.0, "n": 0}
            continue

        r, p = pearsonr(x_clean, y_clean)
        results[feat] = {
            "r": round(float(r), 6),
            "p_value": float(p),
            "abs_r": round(abs(float(r)), 6),
            "n": int(len(x_clean)),
        }

    return results


# ═══════════════════════════════════════════════════════════════
# 1.5.3 Decision Gate
# ═══════════════════════════════════════════════════════════════

def apply_decision_gate(correlations: dict) -> str:
    """Apply the signal existence decision gate.

    Returns:
        "GREEN", "AMBER", or "RED"
    """
    max_abs_r = max(c["abs_r"] for c in correlations.values())

    if max_abs_r > 0.03:
        return "GREEN"
    elif max_abs_r >= 0.02:
        return "AMBER"
    else:
        return "RED"


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Phase 1.5: Signal Existence Checkpoint 1")
    print("=" * 60)

    # Load labels
    labels_path = OUTPUT_DIR / "labels.parquet"
    if not labels_path.exists():
        print("ERROR: labels.parquet not found. Run Phase 1 first.")
        sys.exit(1)

    labels = pd.read_parquet(labels_path)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    print(f"Loaded {len(labels):,} bricks ({(~labels['exclude_flag']).sum():,} resolved)")

    # Extract features
    print("\n📊 Extracting raw features at brick close ticks...")
    features = extract_raw_features(labels)

    # Correlation analysis
    print("\n📈 Point-biserial correlations with y_class:")
    print("-" * 50)
    correlations = run_correlation_analysis(features)

    for feat, result in correlations.items():
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
        print(f"  {feat:15s}  r = {result['r']:+.6f}  |r| = {result['abs_r']:.6f}  "
              f"p = {result['p_value']:.2e}  ({sig})  n = {result['n']:,}")

    # Decision gate
    decision = apply_decision_gate(correlations)

    print("\n" + "=" * 50)
    if decision == "GREEN":
        print("🟢 GREEN — Signal confirmed. At least one raw feature")
        print("   has |r| > 0.03 with y_class. Proceed with confidence.")
    elif decision == "AMBER":
        print("🟡 AMBER — Weak signal detected. Raw features show")
        print("   marginal linear correlation (|r| 0.02–0.03).")
        print("   CNN+LSTM must find non-linear patterns to succeed.")
    else:
        print("🔴 RED — No linear signal detected. All raw features")
        print("   have |r| < 0.02 with y_class. Proceed with caution.")
        print("   The model must rely entirely on non-linear patterns.")
    print("=" * 50)

    # Feature distribution summary
    print("\n📊 Feature distributions:")
    for feat in ["raw_ofi", "raw_velocity", "raw_spread"]:
        vals = features[feat].dropna()
        print(f"  {feat:15s}  mean={vals.mean():.6f}  std={vals.std():.6f}  "
              f"min={vals.min():.6f}  max={vals.max():.6f}")

    # Save results
    output = {
        "checkpoint": "signal_check_1",
        "n_bricks": len(features),
        "correlations": correlations,
        "decision": decision,
        "thresholds": {
            "green": "|r| > 0.03",
            "amber": "0.02 <= |r| <= 0.03",
            "red": "|r| < 0.02",
        },
    }

    out_path = OUTPUT_DIR / "signal_check_1.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved to {out_path}")

    return output


if __name__ == "__main__":
    main()
