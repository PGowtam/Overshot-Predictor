"""Phase 0: Data Audit Script — Verify data availability and print summary stats."""

import pandas as pd
import numpy as np
import os
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "Data" / "Raw"
TICK_DIR = DATA / "Ticks"

def main():
    print("=" * 60)
    print(" Phase 0: Data Audit")
    print("=" * 60)

    # ── 1. Load Renko CSV ──────────────────────────────────────
    renko_path = DATA / "renko_with_tick_outcomes_no_be_XAUUSD20-24.csv"
    print(f"\n📄 Loading Renko CSV: {renko_path.name}")
    df = pd.read_csv(renko_path)
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    assert len(df) > 30000, f"Expected ~30,978 rows, got {len(df)}"

    # ── 2. Summary stats ──────────────────────────────────────
    df['date'] = pd.to_datetime(df['date'])
    print(f"\n📊 Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"   Row count: {len(df):,}")

    outcome_counts = df['outcome'].value_counts()
    total = len(df)
    for outcome, count in outcome_counts.items():
        print(f"   {outcome}: {count:,} ({count/total*100:.1f}%)")

    if 'brick_size' in df.columns:
        bs = df['brick_size']
        print(f"\n📏 Brick size distribution:")
        print(f"   unique values: {sorted(bs.unique())}")
        print(f"   mean: {bs.mean():.4f}, std: {bs.std():.4f}")
        print(f"   min: {bs.min():.4f}, max: {bs.max():.4f}")
    else:
        print("\n⚠️  No 'brick_size' column found in CSV")

    if 'uptrend' in df.columns:
        up = df['uptrend'].sum()
        down = total - up
        print(f"\n📈 Direction: UP={up:,} ({up/total*100:.1f}%), DOWN={down:,} ({down/total*100:.1f}%)")

    # ── 3. Check tick data availability ───────────────────────
    print(f"\n📂 Tick data directory: {TICK_DIR}")
    if not TICK_DIR.exists():
        print("   ❌ TICK DIRECTORY NOT FOUND!")
        return

    years = sorted([d.name for d in TICK_DIR.iterdir() if d.is_dir()])
    print(f"   Years available: {years}")

    total_files = 0
    for year in years:
        year_dir = TICK_DIR / year
        count = sum(1 for _ in year_dir.rglob("*.parquet"))
        total_files += count
        print(f"   {year}: {count} parquet files")

    print(f"   Total tick files: {total_files:,}")

    # ── 4. Verify tick schema on first file ───────────────────
    first_file = TICK_DIR / "2020" / "01" / "01.parquet"
    print(f"\n🔍 Schema check: {first_file.relative_to(BASE)}")
    if first_file.exists():
        tdf = pd.read_parquet(first_file)
        print(f"   Columns: {list(tdf.columns)}")
        print(f"   Shape: {tdf.shape}")
        print(f"   Dtypes:\n{tdf.dtypes.to_string()}")
        print(f"   First 3 rows:")
        print(tdf.head(3).to_string())

        expected_cols = {'timestamp', 'bid', 'bid_vol', 'ask', 'ask_vol'}
        # Check with flexible column naming
        actual_cols = set(tdf.columns)
        missing = expected_cols - actual_cols
        if missing:
            print(f"\n   ⚠️  Missing expected columns: {missing}")
            print(f"   Available columns: {actual_cols}")
        else:
            print(f"\n   ✅ All expected columns present")
    else:
        print(f"   ❌ File not found: {first_file}")

    # ── 5. Spot-check 3 random dates ──────────────────────────
    print(f"\n🎲 Spot-check tick files for 3 random dates:")
    spot_dates = [
        ("2020", "06", "15"),
        ("2021", "09", "22"),
        ("2023", "03", "10"),
    ]

    for year, month, day in spot_dates:
        fpath = TICK_DIR / year / month / f"{day}.parquet"
        if fpath.exists():
            tdf = pd.read_parquet(fpath)
            print(f"   {year}-{month}-{day}: ✅ {len(tdf):,} ticks, "
                  f"bid range [{tdf.iloc[:,1].min():.2f}, {tdf.iloc[:,1].max():.2f}]")
        else:
            print(f"   {year}-{month}-{day}: ❌ Not found")

    print("\n" + "=" * 60)
    print(" Phase 0: Data Audit COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
