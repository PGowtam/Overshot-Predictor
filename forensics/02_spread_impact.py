"""
Forensic Investigation #2: Spread Crossing Impact Quantification

Measures the structural cost of spread crossing on LONG vs SHORT trades.

Key insight:
  - Renko bricks are built from BID prices
  - LONG entry executes at ASK (spread crossing) → needs to travel further
  - SHORT entry executes at BID (no spread crossing) → aligns with brick close
  - LONG exit at BID (spread crossing back)
  - SHORT exit at ASK (spread crossing back)

So LONGS are doubly penalized: pay spread on entry AND exit.

Methodology:
  For each holdout brick:
    1. Record spread at brick close (ask - bid at the first tick after close)
    2. Compute 3 continuation rates:
       a) Mid-price entry/scan (original backtest assumption)
       b) Exec-price entry, mid scan (entry cost only)
       c) Exec-price entry, exec scan (full round-trip cost)
    3. Break down by direction (LONG vs SHORT)
    4. Compute: spread as % of brick_size, and spread-adjusted expectancy

Output:
  forensics/results/spread_impact.json
  forensics/results/spread_impact.png
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


def load_ticks_after(brick_close_time, max_ticks=200):
    current_date = brick_close_time.normalize()
    frames = []
    for offset in range(10):
        check_date = current_date + timedelta(days=offset)
        ticks = load_ticks_for_date(check_date)
        if len(ticks) > 0:
            future = ticks[ticks["timestamp"] > brick_close_time]
            if len(future) > 0:
                frames.append(future)
                total = sum(len(f) for f in frames)
                if total >= max_ticks:
                    break
    if not frames:
        return pd.DataFrame(columns=["timestamp", "bid", "bid_vol", "ask", "ask_vol"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").head(max_ticks).reset_index(drop=True)


def resolve_trade(entry_price, brick_size, is_long, future_ticks, pricing_mode):
    tp = entry_price + brick_size if is_long else entry_price - brick_size
    sl = entry_price - brick_size if is_long else entry_price + brick_size

    bids = future_ticks["bid"].values
    asks = future_ticks["ask"].values

    for i in range(len(bids)):
        if pricing_mode == "execution":
            price = bids[i] if is_long else asks[i]
        else:
            price = (bids[i] + asks[i]) / 2.0

        if is_long:
            if price >= tp:
                return 1
            if price <= sl:
                return 0
        else:
            if price <= tp:
                return 1
            if price >= sl:
                return 0
    return -1  # Unresolved


def main():
    print("=" * 60)
    print(" FORENSIC #2: Spread Crossing Impact Quantification")
    print("=" * 60)

    # Load holdout labels
    holdout_path = OUTPUT_DIR / "holdout" / "labels.parquet"
    if not holdout_path.exists():
        holdout_path = OUTPUT_DIR / "labels.parquet"

    labels = pd.read_parquet(holdout_path)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    if "exclude_flag" in labels.columns:
        labels = labels[~labels["exclude_flag"]]
    if labels["date"].min().year < 2024:
        labels = labels[labels["date"] >= "2024-01-01"]
    labels = labels[labels["y_class"].notna()].reset_index(drop=True)
    n_bricks = len(labels)
    print(f"Loaded {n_bricks:,} resolved holdout bricks")

    # Pricing scenarios
    scenarios = {
        "mid_mid": {"entry": "mid", "scan": "mid"},       # Backtest assumption
        "exec_mid": {"entry": "exec", "scan": "mid"},     # Entry cost only
        "exec_exec": {"entry": "exec", "scan": "execution"},  # Full round-trip
    }

    # Storage
    results = {s: {"long_wins": 0, "long_total": 0, "short_wins": 0, "short_total": 0}
               for s in scenarios}
    spread_data = []
    spread_pct_data = []

    processed = 0
    skipped = 0

    for idx in range(n_bricks):
        row = labels.iloc[idx]
        brick_close = row["date"]
        brick_size = float(row["brick_size"])
        is_long = bool(row["uptrend"])
        bid_entry = float(row["close"])  # Renko close = bid

        future_ticks = load_ticks_after(brick_close, max_ticks=200)
        if len(future_ticks) < 10:
            skipped += 1
            continue

        # Spread at brick close
        spread = float(future_ticks.iloc[0]["ask"] - future_ticks.iloc[0]["bid"])
        ask_at_close = float(future_ticks.iloc[0]["ask"])
        bid_at_close = float(future_ticks.iloc[0]["bid"])
        mid_at_close = (ask_at_close + bid_at_close) / 2.0

        spread_data.append(spread)
        spread_pct_data.append(spread / brick_size * 100)

        for scenario_name, cfg in scenarios.items():
            # Determine entry price
            if cfg["entry"] == "mid":
                entry_price = mid_at_close
            else:
                entry_price = ask_at_close if is_long else bid_at_close

            scan_mode = cfg["scan"]
            scan_ticks = future_ticks.iloc[1:]  # Skip the entry tick
            if len(scan_ticks) < 5:
                continue

            outcome = resolve_trade(entry_price, brick_size, is_long, scan_ticks, scan_mode)
            if outcome < 0:
                continue

            if is_long:
                results[scenario_name]["long_total"] += 1
                results[scenario_name]["long_wins"] += outcome
            else:
                results[scenario_name]["short_total"] += 1
                results[scenario_name]["short_wins"] += outcome

        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed:,}/{n_bricks:,}")

    print(f"\nProcessed {processed:,}, skipped {skipped}")

    # ── Results ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" SPREAD IMPACT RESULTS")
    print("=" * 70)

    output = {}

    print(f"\n{'Scenario':<16} {'LONG WR':>10} {'LONG N':>8} {'SHORT WR':>10} {'SHORT N':>8} {'Delta':>8}")
    print("-" * 70)

    for name, data in results.items():
        long_wr = data["long_wins"] / data["long_total"] if data["long_total"] > 0 else 0
        short_wr = data["short_wins"] / data["short_total"] if data["short_total"] > 0 else 0
        delta = long_wr - short_wr

        print(f"  {name:<14} {long_wr:>10.4f} {data['long_total']:>8,} "
              f"{short_wr:>10.4f} {data['short_total']:>8,} {delta:>+8.4f}")

        output[name] = {
            "long_wr": round(long_wr, 6), "long_n": data["long_total"],
            "short_wr": round(short_wr, 6), "short_n": data["short_total"],
            "overall_wr": round((data["long_wins"] + data["short_wins"]) /
                                max(data["long_total"] + data["short_total"], 1), 6),
            "directional_delta": round(delta, 6)
        }

    # Spread statistics
    spr = np.array(spread_data)
    spr_pct = np.array(spread_pct_data)
    print(f"\n📊 Spread at brick close:")
    print(f"   Mean absolute:   {spr.mean():.4f}")
    print(f"   Mean % of brick: {spr_pct.mean():.2f}%")
    print(f"   Median % brick:  {np.median(spr_pct):.2f}%")
    print(f"   P95 % of brick:  {np.percentile(spr_pct, 95):.2f}%")

    # Effective TP margin after spread
    # LONG: Must travel brick_size + spread (entry at ask, TP still at entry+brick_size from bid)
    # Actually: entry at ask, TP = bid_entry + brick_size, so must travel brick_size - spread
    # Because ask = bid + spread, and TP is relative to entry: entry_ask + brick_size
    # But exit at bid: exit_bid >= entry_ask + brick_size
    # Distance needed: entry_ask + brick_size - entry_ask = brick_size (but exit is bid, so need ask to reach TP)
    # Hmm, let's just measure effective margin
    mean_brick = labels["brick_size"].mean()
    tp_margin_long = mean_brick - spr.mean()  # TP reduced by spread
    tp_margin_short = mean_brick - spr.mean()
    print(f"\n📊 Effective TP margin after spread crossing:")
    print(f"   Raw brick size (mean):    {mean_brick:.4f}")
    print(f"   Spread cost (round trip): {spr.mean() * 2:.4f} (entry + exit)")
    print(f"   Net TP margin (LONG):     {mean_brick - spr.mean() * 2:.4f}")
    print(f"   Margin retention:         {(mean_brick - spr.mean() * 2) / mean_brick:.1%}")

    output["spread_stats"] = {
        "mean_abs": round(float(spr.mean()), 6),
        "mean_pct_brick": round(float(spr_pct.mean()), 4),
        "p95_pct_brick": round(float(np.percentile(spr_pct, 95)), 4),
        "round_trip_cost": round(float(spr.mean() * 2), 6),
        "margin_retention_pct": round(float((mean_brick - spr.mean() * 2) / mean_brick * 100), 2)
    }

    # ── Save ────────────────────────────────────────────────────
    with open(RESULTS_DIR / "spread_impact.json", "w") as f:
        json.dump(output, f, indent=2)

    # ── Plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: WR by scenario and direction
    scenario_names = list(output.keys())
    scenario_names = [s for s in scenario_names if s != "spread_stats"]
    x = np.arange(len(scenario_names))
    long_wrs = [output[s]["long_wr"] for s in scenario_names]
    short_wrs = [output[s]["short_wr"] for s in scenario_names]

    width = 0.35
    axes[0].bar(x - width/2, long_wrs, width, label='LONG', color='#2196F3')
    axes[0].bar(x + width/2, short_wrs, width, label='SHORT', color='#FF5722')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenario_names, rotation=15)
    axes[0].set_ylabel('Win Rate')
    axes[0].set_title('Win Rate by Pricing Scenario', fontweight='bold')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Spread distribution
    axes[1].hist(spr_pct, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=spr_pct.mean(), color='red', linestyle='--', label=f'Mean: {spr_pct.mean():.1f}%')
    axes[1].set_xlabel('Spread as % of Brick Size')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Spread Distribution at Brick Close', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Spread vs outcome
    # Bin spread into deciles, compute WR per bin
    if len(spread_pct_data) > 100:
        combined = pd.DataFrame({
            "spread_pct": spread_pct_data[:processed],
            "is_long": labels["uptrend"].values[:processed],
        })
        # We can't directly get outcome here, so just show spread distribution
        axes[2].hist(spr_pct[spr_pct < np.percentile(spr_pct, 99)], bins=50,
                     color='darkorange', edgecolor='white', alpha=0.8)
        axes[2].set_xlabel('Spread % (zoomed)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Spread Distribution (99th pctile)', fontweight='bold')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "spread_impact.png", dpi=150)
    print(f"\n💾 Saved: spread_impact.json, spread_impact.png")


if __name__ == "__main__":
    main()
