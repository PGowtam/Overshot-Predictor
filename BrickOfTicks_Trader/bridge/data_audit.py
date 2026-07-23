"""
BrickOfTicks Socket Bridge — Broker Data Audit (Phase -1)
==========================================================
Validates that live broker tick data is statistically consistent with
the Dukascopy training data used to train models.

Usage:
  # Collect ticks from live socket bridge (requires EA running):
  python -m bridge.data_audit --collect --duration=60

  # Analyze collected ticks against training data:
  python -m bridge.data_audit --analyze

  # Validate volume fallback proxy:
  python -m bridge.data_audit --validate-fallback

  # Run full audit pipeline:
  python -m bridge.data_audit --full

Acceptance Criteria:
  - Spread drift < 20%    → PASS
  - Velocity drift < 50%  → PASS (WARNING if borderline)
  - Volume available > 50% → Full mode; else Fallback required
  - Fallback OFI balanced within 10% of 0.50 → PASS
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for bridge imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ─── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "bridge" / "logs" / "audit"
TRAINING_TICK_DIR = PROJECT_ROOT.parent / "Data" / "Raw" / "Ticks"


def ensure_audit_dir():
    """Create audit output directory if it doesn't exist."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Phase -1.1: Tick Collection
# ═══════════════════════════════════════════════════════════════════════

def collect_ticks_from_socket(duration_minutes=60, port=9000):
    """
    Collect raw ticks from the socket bridge for the specified duration.
    Saves to bridge/logs/audit/live_ticks_<date>.parquet.

    This requires:
      1. MT5 terminal running with TickSender.mq5 attached
      2. Python tick_receiver started on port 9000

    Returns:
        Path to saved parquet file, or None on failure.
    """
    from bridge.tick_receiver import TickReceiver

    ensure_audit_dir()
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = AUDIT_DIR / f"live_ticks_{date_str}.parquet"

    print(f"\n{'='*60}")
    print(f"  TICK COLLECTION — Phase -1.1")
    print(f"  Duration: {duration_minutes} minutes")
    print(f"  Output:   {output_path}")
    print(f"{'='*60}\n")

    receiver = TickReceiver(port=port)
    receiver.start()

    # Wait for EA to connect and send DAYOPEN
    print("Waiting for EA connection (up to 60s)...")
    start_wait = time.time()
    while receiver.day_open_price is None:
        if time.time() - start_wait > 60:
            print("ERROR: No DAYOPEN received within 60s. Is the EA running?")
            receiver.stop()
            return None
        time.sleep(0.5)

    print(f"Connected! Day open price: {receiver.day_open_price}")

    # Wait for history batch
    print("Waiting for history batch (HDONE)...")
    if not receiver.history_done.wait(timeout=30):
        print("WARNING: No HDONE received. Proceeding with live ticks only.")

    # Collect live ticks
    ticks = []
    deadline = time.time() + duration_minutes * 60
    last_report = time.time()
    print(f"\nCollecting live ticks for {duration_minutes} minutes...")

    while time.time() < deadline:
        try:
            tick = receiver.tick_queue.get(timeout=10)
            ticks.append(tick)

            # Progress report every 30 seconds
            if time.time() - last_report >= 30:
                elapsed = (time.time() - (deadline - duration_minutes * 60)) / 60
                print(f"  [{elapsed:.1f}m / {duration_minutes}m] "
                      f"Collected {len(ticks):,} ticks")
                last_report = time.time()

        except Exception:
            # Queue timeout — no ticks for 10s
            print("  WARNING: No ticks for 10s — market may be closed")
            continue

    receiver.stop()

    if len(ticks) < 100:
        print(f"\nERROR: Only {len(ticks)} ticks collected. Market may be closed.")
        return None

    # Save to parquet
    df = pd.DataFrame(ticks)
    df.to_parquet(output_path, index=False)
    print(f"\n✓ Saved {len(df):,} ticks to {output_path}")
    print(f"  EA reported: {receiver.tick_count} live ticks sent")
    print(f"  Python received: {len(df)} ticks captured")
    if receiver.tick_count > 0:
        capture_rate = len(df) / receiver.tick_count * 100
        print(f"  Capture rate: {capture_rate:.2f}%")

    return output_path


# ═══════════════════════════════════════════════════════════════════════
# Phase -1.2: Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════

def load_training_sample(year=2023, n_files=5, max_ticks=50000):
    """
    Load a random sample from Dukascopy training parquet files.

    Args:
        year: Year directory to load from (default 2023)
        n_files: Number of parquet files to sample from
        max_ticks: Maximum total ticks to return

    Returns:
        pd.DataFrame with columns: [timestamp, bid, ask, bid_vol, ask_vol]
    """
    parquet_dir = TRAINING_TICK_DIR / str(year)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {parquet_dir}")

    files = sorted(parquet_dir.rglob("*.parquet"))
    if len(files) == 0:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    # Sample up to n_files from available files
    selected = files[:min(n_files, len(files))]
    dfs = []
    per_file = max_ticks // len(selected)

    for f in selected:
        try:
            df = pd.read_parquet(f)
            if len(df) > per_file:
                df = df.sample(per_file, random_state=42)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not dfs:
        raise FileNotFoundError("No parquet files could be loaded")

    combined = pd.concat(dfs, ignore_index=True).head(max_ticks)
    # CRITICAL: sort by timestamp to preserve tick ordering for velocity calc
    # (df.sample() shuffles rows, which would make inter-tick intervals meaningless)
    ts_col = 'timestamp' if 'timestamp' in combined.columns else 'time_msc'
    if ts_col in combined.columns:
        combined = combined.sort_values(ts_col).reset_index(drop=True)
    return combined


def load_live_ticks(parquet_path=None):
    """
    Load the most recent live tick parquet from audit directory.

    Args:
        parquet_path: Explicit path. If None, loads most recent.

    Returns:
        pd.DataFrame
    """
    if parquet_path:
        return pd.read_parquet(parquet_path)

    files = sorted(AUDIT_DIR.glob("live_ticks_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No live tick files found in {AUDIT_DIR}. Run --collect first.")
    return pd.read_parquet(files[-1])


def audit_distributions(live_df, train_df):
    """
    Compare live broker ticks against training data distributions.

    Checks:
      1. Spread: mean and std deviation
      2. Tick velocity: median inter-tick interval
      3. Volume availability: percentage of ticks with bid_vol > 0

    Returns:
        dict with results and verdict
    """
    results = {}

    # ─── 1. Spread Distribution ───────────────────────────────────
    # Use RELATIVE spread (spread / bid) to normalize for price level differences.
    # Gold at $4500 will naturally have wider absolute spread than at $2400.
    live_spread_abs = live_df['ask'] - live_df['bid']
    train_spread_abs = train_df['ask'] - train_df['bid']
    
    # Filter out negative/zero spreads (data quality)
    live_spread_abs = live_spread_abs[live_spread_abs > 0]
    train_spread_abs = train_spread_abs[train_spread_abs > 0]
    
    # Relative spread (basis points of bid price)
    live_bid = live_df.loc[live_spread_abs.index, 'bid']
    train_bid = train_df.loc[train_spread_abs.index, 'bid']
    live_spread_rel = live_spread_abs / live_bid * 10000  # in basis points
    train_spread_rel = train_spread_abs / train_bid * 10000

    train_mean_rel = float(train_spread_rel.mean())
    live_mean_rel = float(live_spread_rel.mean())
    spread_drift = abs(live_mean_rel - train_mean_rel) / (train_mean_rel + 1e-10) * 100

    results['spread'] = {
        'live_mean_abs': round(float(live_spread_abs.mean()), 5),
        'live_mean_bps': round(live_mean_rel, 4),
        'live_std_bps': round(float(live_spread_rel.std()), 4),
        'live_median_abs': round(float(live_spread_abs.median()), 5),
        'train_mean_abs': round(float(train_spread_abs.mean()), 5),
        'train_mean_bps': round(train_mean_rel, 4),
        'train_std_bps': round(float(train_spread_rel.std()), 4),
        'train_median_abs': round(float(train_spread_abs.median()), 5),
        'drift_pct': round(float(spread_drift), 2),
        'PASS': bool(spread_drift < 20)
    }

    # ─── 2. Tick Velocity (inter-tick interval) ───────────────────
    if 'time_msc' in live_df.columns:
        live_dt = live_df['time_msc'].diff().dropna()
        live_dt = live_dt[live_dt > 0]  # Filter non-positive intervals
    elif 'timestamp' in live_df.columns:
        # Training data uses 'timestamp' column (datetime)
        live_ts = pd.to_datetime(live_df['timestamp']).astype(int) // 10**6
        live_dt = live_ts.diff().dropna()
        live_dt = live_dt[live_dt > 0]
    else:
        live_dt = pd.Series([0])

    if 'time_msc' in train_df.columns:
        train_dt = train_df['time_msc'].diff().dropna()
        train_dt = train_dt[train_dt > 0]
    elif 'timestamp' in train_df.columns:
        train_ts = pd.to_datetime(train_df['timestamp']).astype(int) // 10**6
        train_dt = train_ts.diff().dropna()
        train_dt = train_dt[train_dt > 0]
    else:
        train_dt = pd.Series([0])

    live_median_dt = float(live_dt.median()) if len(live_dt) > 0 else 0
    train_median_dt = float(train_dt.median()) if len(train_dt) > 0 else 0
    vel_drift = abs(live_median_dt - train_median_dt) / (train_median_dt + 1e-10) * 100

    results['velocity'] = {
        'live_median_dt_ms': round(live_median_dt, 2),
        'live_mean_dt_ms': round(float(live_dt.mean()), 2) if len(live_dt) > 0 else 0,
        'train_median_dt_ms': round(train_median_dt, 2),
        'train_mean_dt_ms': round(float(train_dt.mean()), 2) if len(train_dt) > 0 else 0,
        'drift_pct': round(float(vel_drift), 2),
        'PASS': bool(vel_drift < 50)
    }

    # ─── 3. Volume Availability ───────────────────────────────────
    vol_col = 'bid_vol' if 'bid_vol' in live_df.columns else None
    if vol_col:
        live_has_vol = float((live_df[vol_col] > 0).mean())
    else:
        live_has_vol = 0.0

    train_vol_col = 'bid_vol' if 'bid_vol' in train_df.columns else None
    if train_vol_col:
        train_has_vol = float((train_df[train_vol_col] > 0).mean())
    else:
        train_has_vol = 0.0

    results['volume'] = {
        'live_vol_pct': round(live_has_vol * 100, 2),
        'train_vol_pct': round(train_has_vol * 100, 2),
        'fallback_required': bool(live_has_vol < 0.5)
    }

    # ─── 4. Overall Verdict ──────────────────────────────────────
    spread_ok = bool(results['spread']['PASS'])
    vel_ok = bool(results['velocity']['PASS'])
    fallback = bool(results['volume']['fallback_required'])

    results['verdict'] = {
        'spread_ok': spread_ok,
        'velocity_ok': vel_ok,
        'volume_fallback_required': fallback,
        'PROCEED': bool(spread_ok and vel_ok),
        'NOTES': ("Volume fallback active — expect 88.25% WR (vs 90.3%)"
                  if fallback else "Full volume mode")
    }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase -1.3: Volume Fallback Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_fallback(live_df):
    """
    If broker provides no volume, verify that the price-direction proxy
    produces a balanced OFI signal (roughly 50/50 +1/-1).

    The fallback logic:
      raw_ofi = 1.0 if mid > prev_mid else (-1.0 if mid < prev_mid else 0.0)

    This was validated in ablation study: 88.25% WR (only 1.5% below full volume).

    Returns:
        dict with validation results
    """
    # Check if volume data is actually available
    vol_col = 'bid_vol' if 'bid_vol' in live_df.columns else None
    if vol_col and (live_df[vol_col] > 0).mean() > 0.5:
        return {
            'skip': True,
            'reason': 'Volume available (>50% of ticks have bid_vol > 0) — fallback not needed',
            'PASS': True
        }

    # Compute mid price
    mid = (live_df['bid'] + live_df['ask']) / 2.0
    mid_diff = mid.diff()

    # Compute proxy OFI using sign of mid-price change
    proxy_ofi = np.sign(mid_diff.fillna(0))

    # Count proportions
    total = len(proxy_ofi)
    pos_count = int((proxy_ofi > 0).sum())
    neg_count = int((proxy_ofi < 0).sum())
    zero_count = int((proxy_ofi == 0).sum())

    pos_ratio = pos_count / total if total > 0 else 0
    neg_ratio = neg_count / total if total > 0 else 0
    zero_ratio = zero_count / total if total > 0 else 0

    # Balanced check: pos_ratio should be within 10% of 0.50
    # (excluding zeros, the +1/-1 ratio should be roughly equal)
    non_zero = pos_count + neg_count
    if non_zero > 0:
        pos_of_nonzero = pos_count / non_zero
        balanced = abs(pos_of_nonzero - 0.5) < 0.10
    else:
        pos_of_nonzero = 0
        balanced = False

    result = {
        'skip': False,
        'total_ticks': total,
        'pos_count': pos_count,
        'neg_count': neg_count,
        'zero_count': zero_count,
        'pos_ratio': round(pos_ratio, 4),
        'neg_ratio': round(neg_ratio, 4),
        'zero_ratio': round(zero_ratio, 4),
        'pos_of_nonzero': round(pos_of_nonzero, 4),
        'balanced': balanced,
        'PASS': balanced,
        'note': ("Fallback OFI is balanced — suitable for inference" if balanced
                 else "WARNING: Proxy OFI is skewed — investigate broker data feed")
    }

    return result


# ═══════════════════════════════════════════════════════════════════════
# Phase -1.4: Report Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_audit_report(audit_results, fallback_results, output_path=None):
    """
    Generate a markdown audit report and save to audit directory.

    Returns:
        Path to the generated report.
    """
    ensure_audit_dir()
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = AUDIT_DIR / f"audit_report_{date_str}.md"

    sp = audit_results.get('spread', {})
    vel = audit_results.get('velocity', {})
    vol = audit_results.get('volume', {})
    verdict = audit_results.get('verdict', {})

    lines = [
        "# BrickOfTicks — Broker Data Audit Report",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Status**: {'✅ PASS' if verdict.get('PROCEED') else '❌ FAIL'}",
        "",
        "---",
        "",
        "## 1. Spread Distribution (relative, in basis points of bid)",
        "",
        f"| Metric | Live Broker | Training (Dukascopy) |",
        f"|--------|------------|---------------------|",
        f"| Mean (abs) | {sp.get('live_mean_abs', 'N/A')} pts | {sp.get('train_mean_abs', 'N/A')} pts |",
        f"| Mean (bps) | {sp.get('live_mean_bps', 'N/A')} bps | {sp.get('train_mean_bps', 'N/A')} bps |",
        f"| Std (bps)  | {sp.get('live_std_bps', 'N/A')} bps | {sp.get('train_std_bps', 'N/A')} bps |",
        f"| Drift      | {sp.get('drift_pct', 'N/A')}% | (threshold: <20%) |",
        f"| Result     | {'✅ PASS' if sp.get('PASS') else '❌ FAIL'} | |",
        "",
        "## 2. Tick Velocity",
        "",
        f"| Metric | Live Broker | Training (Dukascopy) |",
        f"|--------|------------|---------------------|",
        f"| Median Δt (ms) | {vel.get('live_median_dt_ms', 'N/A')} | {vel.get('train_median_dt_ms', 'N/A')} |",
        f"| Mean Δt (ms)   | {vel.get('live_mean_dt_ms', 'N/A')} | {vel.get('train_mean_dt_ms', 'N/A')} |",
        f"| Drift          | {vel.get('drift_pct', 'N/A')}% | (threshold: <50%) |",
        f"| Result         | {'✅ PASS' if vel.get('PASS') else '❌ FAIL'} | |",
        "",
        "## 3. Volume Availability",
        "",
        f"| Metric | Live Broker | Training (Dukascopy) |",
        f"|--------|------------|---------------------|",
        f"| Ticks with volume (%) | {vol.get('live_vol_pct', 'N/A')}% | {vol.get('train_vol_pct', 'N/A')}% |",
        f"| Fallback Required     | {'YES' if vol.get('fallback_required') else 'NO'} | |",
        "",
    ]

    # Fallback validation section
    if fallback_results and not fallback_results.get('skip', False):
        fb = fallback_results
        lines.extend([
            "## 4. Volume Fallback Validation",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total ticks | {fb.get('total_ticks', 'N/A')} |",
            f"| Positive OFI (+1) | {fb.get('pos_count', 'N/A')} ({fb.get('pos_ratio', 'N/A')}) |",
            f"| Negative OFI (-1) | {fb.get('neg_count', 'N/A')} ({fb.get('neg_ratio', 'N/A')}) |",
            f"| Zero OFI (0)      | {fb.get('zero_count', 'N/A')} ({fb.get('zero_ratio', 'N/A')}) |",
            f"| +1 of non-zero    | {fb.get('pos_of_nonzero', 'N/A')} (target: 0.40-0.60) |",
            f"| Balanced           | {'✅ YES' if fb.get('balanced') else '❌ NO'} |",
            f"| Result             | {'✅ PASS' if fb.get('PASS') else '❌ FAIL'} |",
            "",
            f"> {fb.get('note', '')}",
            "",
        ])
    elif fallback_results and fallback_results.get('skip'):
        lines.extend([
            "## 4. Volume Fallback Validation",
            "",
            f"> SKIPPED: {fallback_results.get('reason', 'Volume data available')}",
            "",
        ])

    # Overall verdict
    lines.extend([
        "---",
        "",
        "## Verdict",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| Spread drift < 20% | {'✅' if sp.get('PASS') else '❌'} |",
        f"| Velocity drift < 50% | {'✅' if vel.get('PASS') else '❌'} |",
        f"| Volume mode | {'Full' if not vol.get('fallback_required') else 'Fallback (88.25% WR)'} |",
        f"| **Overall** | **{'PROCEED' if verdict.get('PROCEED') else 'BLOCKED'}** |",
        "",
        f"> {verdict.get('NOTES', '')}",
    ])

    report_text = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(report_text)

    # Also save raw results as JSON
    json_path = AUDIT_DIR / f"broker_profile_{date_str}.json"
    combined = {
        'audit': audit_results,
        'fallback': fallback_results,
        'timestamp': datetime.now().isoformat()
    }
    with open(json_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n✓ Audit report saved to {output_path}")
    print(f"✓ Broker profile saved to {json_path}")
    return output_path


def print_audit_summary(audit_results, fallback_results):
    """Print a concise summary to terminal."""
    sp = audit_results.get('spread', {})
    vel = audit_results.get('velocity', {})
    vol = audit_results.get('volume', {})
    verdict = audit_results.get('verdict', {})

    print(f"\n{'='*60}")
    print(f"  BROKER DATA AUDIT — RESULTS")
    print(f"{'='*60}")
    print(f"\n  Spread:   {'✅ PASS' if sp.get('PASS') else '❌ FAIL'} "
          f"(drift: {sp.get('drift_pct', '?')}%, threshold: <20%)")
    print(f"  Velocity: {'✅ PASS' if vel.get('PASS') else '❌ FAIL'} "
          f"(drift: {vel.get('drift_pct', '?')}%, threshold: <50%)")
    print(f"  Volume:   {'Full mode' if not vol.get('fallback_required') else '⚠️  Fallback required'} "
          f"({vol.get('live_vol_pct', '?')}% ticks have volume)")

    if fallback_results and not fallback_results.get('skip', False):
        print(f"  Fallback: {'✅ PASS' if fallback_results.get('PASS') else '❌ FAIL'} "
              f"(balance: {fallback_results.get('pos_of_nonzero', '?')})")

    overall = verdict.get('PROCEED', False)
    print(f"\n  {'='*56}")
    print(f"  VERDICT: {'✅ PROCEED TO PHASE 0' if overall else '❌ BLOCKED — DO NOT TRADE'}")
    print(f"  {verdict.get('NOTES', '')}")
    print(f"  {'='*56}\n")


# ═══════════════════════════════════════════════════════════════════════
# Offline Audit (no socket needed — uses existing parquets)
# ═══════════════════════════════════════════════════════════════════════

def run_offline_audit(live_parquet_path=None):
    """
    Run the full audit against already-collected live tick data.
    Does not require a live socket connection.
    """
    print("Loading live tick data...")
    live_df = load_live_ticks(live_parquet_path)
    print(f"  Loaded {len(live_df):,} live ticks")

    print("Loading training tick data (Dukascopy 2023)...")
    train_df = load_training_sample(year=2023, n_files=5, max_ticks=50000)
    print(f"  Loaded {len(train_df):,} training ticks")

    print("\nRunning distribution analysis...")
    audit_results = audit_distributions(live_df, train_df)

    print("Running volume fallback validation...")
    fallback_results = validate_fallback(live_df)

    print_audit_summary(audit_results, fallback_results)
    report_path = generate_audit_report(audit_results, fallback_results)

    return audit_results, fallback_results


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BrickOfTicks Broker Data Audit (Phase -1)")
    parser.add_argument('--collect', action='store_true',
                        help='Collect live ticks from socket bridge')
    parser.add_argument('--duration', type=int, default=60,
                        help='Collection duration in minutes (default: 60)')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze collected ticks vs training data')
    parser.add_argument('--validate-fallback', action='store_true',
                        help='Validate volume fallback proxy')
    parser.add_argument('--full', action='store_true',
                        help='Run full audit pipeline (collect + analyze + validate)')
    parser.add_argument('--live-file', type=str, default=None,
                        help='Path to live tick parquet (for offline analysis)')
    parser.add_argument('--port', type=int, default=9000,
                        help='TCP port for tick collection (default: 9000)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    if args.full:
        # Full pipeline: collect → analyze → validate → report
        parquet_path = collect_ticks_from_socket(
            duration_minutes=args.duration, port=args.port)
        if parquet_path is None:
            print("FAILED: Tick collection failed. Aborting audit.")
            sys.exit(1)
        run_offline_audit(parquet_path)
        return

    if args.collect:
        parquet_path = collect_ticks_from_socket(
            duration_minutes=args.duration, port=args.port)
        if parquet_path is None:
            sys.exit(1)
        return

    if args.analyze or args.validate_fallback:
        run_offline_audit(args.live_file)
        return

    # If no arguments, print help
    parser.print_help()


if __name__ == '__main__':
    main()
