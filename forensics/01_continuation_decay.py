"""
Forensic Investigation #1: Tick-Level Continuation Decay Analysis

Measures: At each tick offset after brick close, what is the probability
that the trade still resolves as a WIN (continuation)?

This directly tests the hypothesis: "The edge is consumed before the
order can be placed because ticks after brick close move price away
from the theoretical entry."

Methodology:
  For each holdout brick:
    1. Load the brick close timestamp and direction
    2. Load the next 20 ticks after brick close
    3. For each tick offset k in [0, 1, 2, 3, 5, 10, 20]:
       - Treat tick[k] as the actual entry point
       - Compute entry price = ask (for LONG) or bid (for SHORT)
       - Scan forward from tick[k+1] for TP/SL resolution
       - Record: win/loss, effective spread cost, latency_ms
    4. Aggregate: P(win) at each offset, with confidence intervals

Output:
  forensics/results/continuation_decay.json
  forensics/results/continuation_decay.png
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "Raw"
TICK_DIR = DATA_DIR / "Ticks"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_ticks_for_date(date):
    path = TICK_DIR / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "bid", "bid_vol", "ask", "ask_vol"])
    df = pd.read_parquet(path)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def load_ticks_after(brick_close_time, max_ticks=50):
    """Load ticks starting from brick close, spanning multiple days if needed."""
    current_date = brick_close_time.normalize()
    frames = []
    days_checked = 0

    for offset in range(10):  # Check up to 10 calendar days
        check_date = current_date + timedelta(days=offset)
        ticks = load_ticks_for_date(check_date)
        if len(ticks) > 0:
            future = ticks[ticks["timestamp"] > brick_close_time]
            if len(future) > 0:
                frames.append(future)
                total = sum(len(f) for f in frames)
                if total >= max_ticks:
                    break
            days_checked += 1
            if days_checked >= 3:
                break

    if not frames:
        return pd.DataFrame(columns=["timestamp", "bid", "bid_vol", "ask", "ask_vol"])
    result = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return result.head(max_ticks)


def resolve_trade(entry_price, brick_size, is_long, future_ticks, pricing_mode="execution"):
    """Scan ticks to determine if trade hits TP or SL first."""
    tp = entry_price + brick_size if is_long else entry_price - brick_size
    sl = entry_price - brick_size if is_long else entry_price + brick_size

    bids = future_ticks["bid"].values
    asks = future_ticks["ask"].values

    for i in range(len(bids)):
        if pricing_mode == "execution":
            scan_price = bids[i] if is_long else asks[i]
        else:
            scan_price = (bids[i] + asks[i]) / 2.0

        if is_long:
            if scan_price >= tp:
                return 1, i  # WIN
            if scan_price <= sl:
                return 0, i  # LOSS
        else:
            if scan_price <= tp:
                return 1, i  # WIN
            if scan_price >= sl:
                return 0, i  # LOSS

    return -1, len(bids)  # Unresolved


def main():
    print("=" * 60)
    print(" FORENSIC #1: Tick-Level Continuation Decay")
    print("=" * 60)

    # Load holdout labels
    holdout_path = OUTPUT_DIR / "holdout" / "labels.parquet"
    if not holdout_path.exists():
        holdout_path = OUTPUT_DIR / "labels.parquet"

    labels = pd.read_parquet(holdout_path)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)

    # Filter to holdout period and resolved bricks
    if "exclude_flag" in labels.columns:
        labels = labels[~labels["exclude_flag"]]
    if labels["date"].min().year < 2024:
        labels = labels[labels["date"] >= "2024-01-01"]

    labels = labels[labels["y_class"].notna()].reset_index(drop=True)
    n_bricks = len(labels)
    print(f"Loaded {n_bricks:,} resolved holdout bricks")

    # Tick offsets to test
    offsets = [0, 1, 2, 3, 5, 10, 15, 20]

    # Storage: offset -> list of (win/loss, latency_ms, spread_at_entry)
    results_by_offset = {k: [] for k in offsets}
    latencies_by_offset = {k: [] for k in offsets}
    spreads_at_close = []

    tick_cache = {}
    processed = 0
    skipped = 0

    for idx in range(n_bricks):
        row = labels.iloc[idx]
        brick_close = row["date"]
        brick_size = float(row["brick_size"])
        is_long = bool(row["uptrend"])
        theoretical_entry = float(row["close"])  # bid-based entry

        # Load ticks after brick close
        future_ticks = load_ticks_after(brick_close, max_ticks=60)

        if len(future_ticks) < 25:
            skipped += 1
            continue

        # Record spread at close (first tick after close)
        spread_at_close = float(future_ticks.iloc[0]["ask"] - future_ticks.iloc[0]["bid"])
        spreads_at_close.append(spread_at_close)

        # Compute brick_close timestamp for latency measurement
        t0 = future_ticks.iloc[0]["timestamp"]

        for offset in offsets:
            if offset >= len(future_ticks) - 5:
                continue  # Not enough ticks after offset for resolution

            entry_tick = future_ticks.iloc[offset]

            # Execution-realistic entry price
            if is_long:
                entry_price = float(entry_tick["ask"])  # Buy at ask
            else:
                entry_price = float(entry_tick["bid"])  # Sell at bid

            # Latency from brick close to entry tick
            latency_ms = (entry_tick["timestamp"] - t0).total_seconds() * 1000.0
            if offset == 0:
                latency_ms = 0.0  # t+0 = the first tick after close

            # Resolve from ticks AFTER the entry tick
            scan_ticks = future_ticks.iloc[offset + 1:]
            if len(scan_ticks) < 3:
                continue

            outcome, _ = resolve_trade(entry_price, brick_size, is_long, scan_ticks, "execution")

            if outcome >= 0:  # Resolved
                results_by_offset[offset].append(outcome)
                latencies_by_offset[offset].append(latency_ms)

        processed += 1
        if (processed) % 500 == 0:
            print(f"  Processed {processed:,}/{n_bricks:,} bricks (skipped {skipped})")

    print(f"\nProcessed {processed:,} bricks, skipped {skipped}")

    # ── Compute statistics ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(" CONTINUATION DECAY RESULTS")
    print("=" * 60)
    print(f"\n{'Offset':>8} {'N':>8} {'WinRate':>10} {'95% CI':>16} {'Med Latency':>14}")
    print("-" * 60)

    decay_data = {}
    for offset in offsets:
        outcomes = results_by_offset[offset]
        n = len(outcomes)
        if n == 0:
            continue

        wr = np.mean(outcomes)
        # Wilson score interval for binomial proportion
        z = 1.96
        denom = 1 + z**2 / n
        centre = (wr + z**2 / (2*n)) / denom
        spread = z * np.sqrt((wr * (1 - wr) + z**2 / (4*n)) / n) / denom
        ci_lo = max(0, centre - spread)
        ci_hi = min(1, centre + spread)

        lats = latencies_by_offset[offset]
        med_lat = np.median(lats) if lats else 0

        print(f"  t+{offset:<5} {n:>8,} {wr:>10.4f} [{ci_lo:.4f}, {ci_hi:.4f}] {med_lat:>12.1f}ms")

        decay_data[f"t+{offset}"] = {
            "n": n, "win_rate": round(wr, 6),
            "ci_lo": round(ci_lo, 6), "ci_hi": round(ci_hi, 6),
            "median_latency_ms": round(med_lat, 2)
        }

    # Spread stats
    if spreads_at_close:
        spr = np.array(spreads_at_close)
        brick_sizes = labels["brick_size"].values[:len(spr)]
        spread_pct = spr / brick_sizes[:len(spr)] * 100
        print(f"\n📊 Spread at brick close:")
        print(f"   Mean: {spr.mean():.4f} ({spread_pct.mean():.1f}% of brick_size)")
        print(f"   Median: {np.median(spr):.4f}")
        print(f"   P95: {np.percentile(spr, 95):.4f}")
        print(f"   Max: {spr.max():.4f}")

        decay_data["spread_stats"] = {
            "mean": round(float(spr.mean()), 6),
            "median": round(float(np.median(spr)), 6),
            "p95": round(float(np.percentile(spr, 95)), 6),
            "mean_pct_of_brick": round(float(spread_pct.mean()), 4)
        }

    # ── Save results ────────────────────────────────────────────
    with open(RESULTS_DIR / "continuation_decay.json", "w") as f:
        json.dump(decay_data, f, indent=2)

    # ── Plot ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_offsets = []
    plot_wrs = []
    plot_ci_lo = []
    plot_ci_hi = []
    plot_lats = []

    for offset in offsets:
        key = f"t+{offset}"
        if key in decay_data and "win_rate" in decay_data[key]:
            plot_offsets.append(offset)
            plot_wrs.append(decay_data[key]["win_rate"])
            plot_ci_lo.append(decay_data[key]["ci_lo"])
            plot_ci_hi.append(decay_data[key]["ci_hi"])
            plot_lats.append(decay_data[key]["median_latency_ms"])

    if plot_offsets:
        # Left: WR vs tick offset
        ax1.fill_between(plot_offsets, plot_ci_lo, plot_ci_hi, alpha=0.2, color='steelblue')
        ax1.plot(plot_offsets, plot_wrs, 'o-', color='steelblue', linewidth=2, markersize=8)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Break-even (50%)')
        ax1.set_xlabel('Tick Offset After Brick Close', fontsize=12)
        ax1.set_ylabel('Continuation Win Rate', fontsize=12)
        ax1.set_title('Edge Decay: P(Continuation) vs Entry Delay', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.35, 0.7)

        # Right: WR vs latency (ms)
        ax2.plot(plot_lats, plot_wrs, 's-', color='darkorange', linewidth=2, markersize=8)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Break-even')
        ax2.set_xlabel('Median Latency from Brick Close (ms)', fontsize=12)
        ax2.set_ylabel('Continuation Win Rate', fontsize=12)
        ax2.set_title('Edge Decay: P(Continuation) vs Latency', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.35, 0.7)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "continuation_decay.png", dpi=150)
    print(f"\n💾 Saved: continuation_decay.json, continuation_decay.png")


if __name__ == "__main__":
    main()
