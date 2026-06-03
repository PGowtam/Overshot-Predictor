"""
Debug: Manual single-day walkthrough of V4 regime logic.
Verifies every stage of the pipeline produces sane values.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, date

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "BrickOfTicks_Trader"))
sys.path.insert(0, str(BASE_DIR / "src"))

import bridge.renko
import bridge.path_optimizer
bridge.renko.K_MULTIPLIER = 0.00118
bridge.path_optimizer.K_MULTIPLIER = 0.00118

from bridge.renko import RenkoBuilder
from bridge.feature_engine import LiveFeatureEngine
from bridge.path_optimizer import PathOptimizer
from regime_tracker_v4 import RegimeTrackerV4

TARGET_DAY = date(2026, 2, 2)  # Known active day from historical study

def main():
    parquet_path = str(BASE_DIR / "Data" / "xauusd_ticks_2026.parquet")
    print(f"Loading ticks from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date

    day = TARGET_DAY
    day_df = df[df['utc_day'] == day]
    print(f"\n{'='*60}")
    print(f" TARGET DAY: {day}")
    print(f" Day ticks: {len(day_df)}")
    print(f"{'='*60}")

    if len(day_df) < 100:
        print("ABORT: Not enough ticks for this day.")
        return

    # --- Stage 1: Build lookback ---
    start_day = day - timedelta(days=7)
    lb_mask = (df['utc_day'] >= start_day) & (df['utc_day'] < day)
    lb_df = df[lb_mask]
    lookback_ticks = lb_df.to_dict('records')
    day_ticks = day_df.to_dict('records')
    print(f"\n[Stage 1] Lookback window: {start_day} to {day}")
    print(f"  Lookback ticks: {len(lookback_ticks)}")
    lb_days = lb_df['utc_day'].unique()
    print(f"  Lookback trading days: {sorted(lb_days)}")

    # --- Stage 2: PathOptimizer ---
    day_open = day_ticks[0]['bid']
    brick_size = day_open * bridge.renko.K_MULTIPLIER
    print(f"\n[Stage 2] Day open: {day_open:.5f}, Brick size: {brick_size:.5f}")

    optimizer = PathOptimizer()
    if len(lookback_ticks) > 1000:
        best_price, best_idx, _ = optimizer.find_optimal_anchor(lookback_ticks, brick_size)
        if best_price is None: best_price = day_open; best_idx = 0
    else:
        best_price = day_open; best_idx = 0
    print(f"  Best anchor: {best_price:.5f}, Start index: {best_idx}")

    # --- Stage 3: Warmup replay ---
    renko = RenkoBuilder(best_price)
    renko.update_brick_size(brick_size, new_day_open=best_price)
    feature_engine = LiveFeatureEngine()
    feature_engine.update_brick_size(brick_size)

    warmup_records = []
    ofi_peak = 0.0
    warmup_bricks = 0
    for i, tick in enumerate(lookback_ticks):
        feat = feature_engine.compute_vector(tick['bid'], tick['ask'], 0.0, 0.0, tick['time_msc'])
        if feat is not None:
            if abs(feat[0]) > abs(ofi_peak): ofi_peak = feat[0]

        if i >= best_idx:
            new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
            for brick in new_bricks:
                warmup_bricks += 1
                feature_engine.on_new_brick(brick)
                t_day = pd.to_datetime(brick.timestamp, unit='ms', utc=True).date()
                if t_day >= (day - timedelta(days=5)) and t_day < day:
                    warmup_records.append({
                        'utc_day': t_day,
                        'spread_current': tick['ask'] - tick['bid'],
                        'abs_ofi_peak': abs(ofi_peak)
                    })
                ofi_peak = 0.0

    print(f"\n[Stage 3] Warmup replay complete")
    print(f"  Warmup bricks formed: {warmup_bricks}")
    print(f"  Warmup records (for tracker): {len(warmup_records)}")

    if warmup_records:
        wr_df = pd.DataFrame(warmup_records)
        print(f"  Spread range: [{wr_df['spread_current'].min():.4f}, {wr_df['spread_current'].max():.4f}]")
        print(f"  OFI range:    [{wr_df['abs_ofi_peak'].min():.4f}, {wr_df['abs_ofi_peak'].max():.4f}]")
        print(f"  Days in warmup: {sorted(wr_df['utc_day'].unique())}")

    # --- Stage 4: RegimeTracker ---
    tracker = RegimeTrackerV4(lookback_days=5)
    if warmup_records:
        df_hist = pd.DataFrame(warmup_records)
        tracker.refresh(day, df_hist)

    print(f"\n[Stage 4] RegimeTrackerV4 state")
    for k, v in tracker.histories.items():
        print(f"  {k}: {len(v)} samples")
    ready = tracker.is_ready(min_samples=100)
    print(f"  is_ready(100): {ready}")

    if not ready:
        print("\n  *** TRACKER NOT READY - this is why no trades are generated! ***")
        print("  Try with lower min_samples:")
        for n in [50, 30, 10, 5]:
            print(f"    is_ready({n}): {tracker.is_ready(min_samples=n)}")
        return

    # --- Stage 5: Live day simulation ---
    print(f"\n[Stage 5] Simulating live day {day}")
    ofi_peak = 0.0
    signals = []
    brick_count = 0

    for tick in day_ticks:
        bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']
        feat = feature_engine.compute_vector(bid, ask, 0.0, 0.0, t_msc)
        if feat is not None:
            if abs(feat[0]) > abs(ofi_peak): ofi_peak = feat[0]

        new_bricks = renko.update_tick(bid, t_msc)
        for brick in new_bricks:
            brick_count += 1
            feature_engine.on_new_brick(brick)

            spread_current = ask - bid
            abs_ofi = abs(ofi_peak)

            sp_pct = tracker.get_percentile('spread_current', spread_current)
            op_pct = tracker.get_percentile('abs_ofi_peak', abs_ofi)

            brick_dt = pd.to_datetime(brick.timestamp, unit='ms', utc=True)

            # Log first 10 bricks regardless
            if brick_count <= 10:
                print(f"  Brick {brick_count}: spread={spread_current:.4f} (pct={sp_pct:.1f}%), "
                      f"ofi={abs_ofi:.4f} (pct={op_pct:.1f}%), hour={brick_dt.hour}")

            if sp_pct >= 95 and op_pct <= 20:
                signals.append({
                    'time': str(brick_dt),
                    'spread_raw': spread_current,
                    'spread_pct': sp_pct,
                    'ofi_raw': abs_ofi,
                    'ofi_pct': op_pct,
                    'hour': brick_dt.hour,
                    'direction': brick.uptrend
                })

            ofi_peak = 0.0

    print(f"\n[Stage 5 Results]")
    print(f"  Total bricks on {day}: {brick_count}")
    print(f"  Total signals (95/20): {len(signals)}")

    if signals:
        print(f"\n  First 5 signals:")
        for s in signals[:5]:
            print(f"    {s['time']} | spread={s['spread_raw']:.4f} ({s['spread_pct']:.1f}%) "
                  f"| ofi={s['ofi_raw']:.4f} ({s['ofi_pct']:.1f}%) | hr={s['hour']} | dir={s['direction']}")
    else:
        print("\n  *** NO SIGNALS FIRED ***")
        print("  Checking percentile distribution of today's bricks...")

if __name__ == "__main__":
    main()
