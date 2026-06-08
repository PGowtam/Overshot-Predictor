"""
sc_sim_2026_main.py
Corrected live simulation on the FULL 2026 main tick dataset.
Bugs fixed vs previous version:
  1. scan_exit_ticks starts at fill_index+1 (no instant ghost wins)
  2. last_signal_brick_time guard prevents cascading trades on same-timestamp bricks
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Simulator2026Main")

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
MTPATSC_DIR = BASE_DIR / "MTPATSC"
OUTPUT_DIR  = MTPATSC_DIR / "outputs" / "setup_classifier"
SIM_OUT_DIR = OUTPUT_DIR / "sim_2026_corrected"


def scan_exit_ticks(bids, asks, times, start_idx, direction, entry, tp, sl):
    """
    Scan tick-by-tick for TP/SL starting from start_idx + 1.
    Skipping the fill tick itself prevents instant ghost wins where price 
    has already moved past TP at the moment of entry.
    Returns (is_win, exit_time).
    """
    for i in range(start_idx + 1, len(bids)):
        bid = bids[i]
        ask = asks[i]
        t   = times[i]
        if direction == 1:  # Long
            if bid >= tp:   return True,  t
            if bid <= sl:   return False, t
        else:               # Short
            if ask <= tp:   return True,  t
            if ask >= sl:   return False, t
    return False, times[-1]


def load_ticks_safe(path: Path) -> pd.DataFrame:
    """Read a tick parquet row-group by row-group to survive histogram corruption."""
    f = pq.ParquetFile(str(path))
    dfs = []
    for i in range(f.metadata.num_row_groups):
        try:
            dfs.append(f.read_row_group(i).to_pandas())
        except Exception as e:
            logger.warning(f"Skipping row group {i}: {e}")
    if not dfs:
        raise RuntimeError(f"Could not read any row groups from {path}")
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df):,} ticks from {path.name}")
    return df


def main():
    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ticks ────────────────────────────────────────────────────────
    ticks_path = BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"
    if not ticks_path.exists():
        ticks_path = BASE_DIR / "data" / "xauusd_ticks_2026.parquet"
    if not ticks_path.exists():
        logger.error(f"Ticks not found at {ticks_path}")
        return

    df_ticks = load_ticks_safe(ticks_path)
    if 'time_msc' not in df_ticks.columns and 'timestamp' in df_ticks.columns:
        df_ticks['time_msc'] = pd.to_datetime(df_ticks['timestamp']).astype('int64') // 10**6
    df_ticks = df_ticks.sort_values('time_msc').reset_index(drop=True)

    bids  = np.ascontiguousarray(df_ticks['bid'].values,      dtype=np.float64)
    asks  = np.ascontiguousarray(df_ticks['ask'].values,      dtype=np.float64)
    times = np.ascontiguousarray(df_ticks['time_msc'].values, dtype=np.int64)
    logger.info(f"Tick range: {datetime.utcfromtimestamp(times[0]/1000)} -> {datetime.utcfromtimestamp(times[-1]/1000)}")

    # ── 2. Load bricks (use cached parquet from extractor) ───────────────────
    bricks_path = OUTPUT_DIR / "sim_2026_bricks.parquet"
    if not bricks_path.exists():
        logger.error(f"Bricks parquet not found: {bricks_path}  -- run sc_sim_extractor.py first")
        return

    logger.info(f"Loading bricks from {bricks_path}...")
    df_bricks = load_ticks_safe(bricks_path)   # same safe reader
    logger.info(f"Loaded {len(df_bricks)} bricks")

    bricks_data = []
    for _, row in df_bricks.iterrows():
        bricks_data.append({
            "brick_id":    int(row["brick_id"]),
            "timestamp":   int(row["timestamp"]),
            "direction":   int(row["direction"]),
            "close_price": float(row["close_price"]),
            "open_price":  float(row["open_price"]),
            "brick_size":  float(row["brick_size"]),
            "t1_win":      int(row.get("t1_win", 0)),
            "t2_win":      int(row.get("t2_win", 0)),
            "t3_win":      int(row.get("t3_win", 0)),
            "t4_win":      int(row.get("t4_win", 0)),
            "label":       int(row.get("label", 0)),
            "ancs_fine":   np.stack(row["ancs_fine"]).astype(np.float32),
            "ancs_coarse": np.stack(row["ancs_coarse"]).astype(np.float32),
            "history":     np.stack([np.stack(x) for x in row["history"]]).astype(np.float32),
            "candle_features": np.array(row["candle_features"], dtype=np.float32),
            "momentum":    np.array(row["momentum"],        dtype=np.float32),
        })

    # ── 3. Load cached probabilities ─────────────────────────────────────────
    probs_path = OUTPUT_DIR / "sim_2026_probs.npy"
    if not probs_path.exists():
        logger.error(f"Probs .npy not found: {probs_path}  -- run sc_sim_predictor.py first")
        return

    probs       = np.load(str(probs_path))
    pred_classes = np.argmax(probs, axis=1)
    logger.info(f"Probabilities loaded: {probs.shape}")

    # ── 4. Load config ────────────────────────────────────────────────────────
    with open(OUTPUT_DIR / "config.json") as f:
        config = json.load(f)

    T1_threshold   = config.get("T1_threshold",   1.0)
    T2_threshold   = config.get("T2_threshold",   1.0)
    T3_threshold   = config.get("T3_threshold",   1.0)
    T4_threshold   = config.get("T4_threshold",   1.0)
    veto_threshold = config.get("T0_veto_threshold", 0.40)
    thresholds  = {1: T1_threshold, 2: T2_threshold, 3: T3_threshold, 4: T4_threshold}
    rr_profiles = {1: 1.0,          2: 2.0,          3: 2.0,          4: 3.0}

    logger.info(f"Config: T1={T1_threshold} T2={T2_threshold} T3={T3_threshold} T4={T4_threshold} Veto={veto_threshold}")

    # ── 5. Threshold winrate table (on brick labels, not live ticks) ──────────
    logger.info("\n==================================================")
    logger.info(" THEORETICAL WINRATES AT MULTIPLE THRESHOLDS")
    logger.info("==================================================")
    thresholds_to_test = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    win_flags = {
        1: np.array([b['t1_win'] for b in bricks_data]),
        2: np.array([b['t2_win'] for b in bricks_data]),
        3: np.array([b['t3_win'] for b in bricks_data]),
        4: np.array([b['t4_win'] for b in bricks_data]),
    }
    thresholds_table_md = "### Theoretical Winrate (labels from C++ engine)\n\n"
    for sc in [1, 2, 3, 4]:
        logger.info(f"\nSetup T{sc}:")
        logger.info(f"  {'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10}")
        logger.info("  " + "-" * 34)
        thresholds_table_md += f"#### Setup T{sc}\n\n| Threshold | Trades | Win Rate |\n| :--- | :--- | :--- |\n"
        t_win = win_flags[sc]
        for th in thresholds_to_test:
            mask = (pred_classes == sc) & (probs[:, sc] >= th) & (probs[:, 0] <= veto_threshold)
            n = int(mask.sum())
            if n > 0:
                wr = float(np.mean(t_win[mask]))
                logger.info(f"  {th:<10.2f} | {n:<8} | {wr:<10.2%}")
                thresholds_table_md += f"| {th:.2f} | {n} | {wr:.2%} |\n"
            else:
                logger.info(f"  {th:<10.2f} | {'0':<8} | {'N/A':<10}")
                thresholds_table_md += f"| {th:.2f} | 0 | N/A |\n"
        thresholds_table_md += "\n"

    # ── 6. Live simulation loop ───────────────────────────────────────────────
    active_trade          = None
    trade_log             = []
    daily_summaries       = {}
    last_signal_brick_time = -1   # Guard: no cascade on same-timestamp bricks

    logger.info("Starting corrected live simulation loop...")
    for i, brick in enumerate(bricks_data):
        b_time   = brick['timestamp']
        date_str = datetime.utcfromtimestamp(b_time / 1000.0).date().strftime("%Y-%m-%d")

        if date_str not in daily_summaries:
            daily_summaries[date_str] = {
                "date": date_str, "trades": 0, "wins": 0, "losses": 0,
                "daily_pnl_R": 0.0, "bricks_count": 0
            }
        daily_summaries[date_str]["bricks_count"] += 1

        # Close active trade when a newer brick arrives
        if active_trade is not None and b_time >= active_trade['exit_time']:
            trade_log.append(active_trade)
            ed = datetime.utcfromtimestamp(active_trade['entry_time'] / 1000.0).date().strftime("%Y-%m-%d")
            daily_summaries[ed]["trades"] += 1
            if active_trade['pnl_R'] > 0:
                daily_summaries[ed]["wins"]  += 1
            else:
                daily_summaries[ed]["losses"] += 1
            daily_summaries[ed]["daily_pnl_R"] += active_trade['pnl_R']
            active_trade = None

        # Daily drawdown stop
        if daily_summaries[date_str]["daily_pnl_R"] <= -5.0:
            continue

        # Signal check
        if active_trade is not None:
            continue
        # Timestamp collision guard
        if b_time == last_signal_brick_time:
            continue

        p_t0 = probs[i, 0]
        if p_t0 > veto_threshold:
            continue

        pred_c = pred_classes[i]
        if pred_c not in thresholds:
            continue
        theta = thresholds[pred_c]
        if theta >= 1.0:
            continue
        if probs[i, pred_c] < theta:
            continue

        # Entry
        tick_idx = np.searchsorted(times, b_time)
        if tick_idx >= len(times):
            continue

        direction   = brick['direction']
        if direction == 0:
            direction = -1

        bs          = brick['brick_size']
        rr          = rr_profiles[pred_c]
        close_price = brick['close_price']
        bid_start   = bids[tick_idx]
        ask_start   = asks[tick_idx]
        spread      = ask_start - bid_start
        fill_index  = tick_idx

        if pred_c == 1:  # T1 Continuation
            if direction == 1:
                entry = close_price + spread
                tp    = entry + bs
                sl    = entry - bs
            else:
                entry = close_price - spread
                tp    = entry - bs
                sl    = entry + bs

        elif pred_c == 2:  # T2 Limit at Open
            limit_price = brick['open_price']
            filled = False
            for j in range(tick_idx, len(times)):
                if direction == 1 and asks[j] <= limit_price:
                    filled = True; fill_index = j; break
                elif direction == -1 and bids[j] >= limit_price:
                    filled = True; fill_index = j; break
                limit_sl = limit_price - (direction * bs)
                tick_p   = bids[j] if direction == 1 else asks[j]
                if (direction == 1 and tick_p <= limit_sl) or (direction == -1 and tick_p >= limit_sl):
                    break
            if not filled:
                continue
            entry = limit_price
            tp    = entry + (direction * 2.0 * bs)
            sl    = entry - (direction * bs)

        elif pred_c == 3:  # T3 Reversal
            entry     = asks[tick_idx] if direction == -1 else bids[tick_idx]
            tp        = entry - (direction * 2.0 * bs)
            sl        = entry + (direction * bs)
            direction = -direction

        elif pred_c == 4:  # T4 Deep Reversal
            entry     = asks[tick_idx] if direction == -1 else bids[tick_idx]
            tp        = entry - (direction * 3.0 * bs)
            sl        = entry + (direction * bs)
            direction = -direction

        else:
            continue

        is_win, exit_time = scan_exit_ticks(bids, asks, times, fill_index, direction, entry, tp, sl)
        pnl_R = rr if is_win else -1.0

        last_signal_brick_time = b_time

        active_trade = {
            "brick_id":    brick['brick_id'],
            "setup_type":  f"T{pred_c}",
            "direction":   "BUY" if direction == 1 else "SELL",
            "entry_price": round(entry, 2),
            "sl":          round(sl,    2),
            "tp":          round(tp,    2),
            "entry_time":  times[fill_index],
            "exit_time":   exit_time,
            "pnl_R":       pnl_R,
        }

    if active_trade is not None:
        trade_log.append(active_trade)

    # ── 7. Reports ────────────────────────────────────────────────────────────
    trades_df = pd.DataFrame(trade_log)
    trades_df.to_csv(SIM_OUT_DIR / "sim_trades.csv", index=False)

    daily_df = pd.DataFrame(list(daily_summaries.values()))
    daily_df.to_csv(SIM_OUT_DIR / "sim_daily.csv", index=False)

    total_trades = len(trade_log)
    wins         = sum(1 for t in trade_log if t['pnl_R'] > 0)
    losses       = total_trades - wins
    win_rate     = wins / total_trades if total_trades else 0.0
    total_R      = sum(t['pnl_R'] for t in trade_log)

    # Equity curve
    if total_trades > 0:
        cum = np.cumsum([t['pnl_R'] for t in trade_log])
        plt.figure(figsize=(12, 5))
        plt.plot(cum, color="steelblue", linewidth=2)
        plt.axhline(0, color="red", linestyle="--", alpha=0.4)
        plt.title(f"2026 Corrected Simulation  |  WR {win_rate:.2%}  |  Net {total_R:+.1f} R")
        plt.xlabel("Trade #"); plt.ylabel("Cumulative R")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(SIM_OUT_DIR / "sim_equity.png", dpi=150)
        plt.close()

    # Ghost trade sanity check
    ghost_count = sum(1 for t in trade_log if t['entry_time'] == t['exit_time'])

    report = f"""# 2026 Corrected Live Simulation Report

## Bug Fixes Applied
- `scan_exit_ticks` now starts at `fill_index + 1` (no instant wins)
- `last_signal_brick_time` guard prevents same-timestamp brick cascades

## Config
| Parameter | Value |
| :--- | :--- |
| T1 Threshold | {T1_threshold} |
| T2 Threshold | {T2_threshold} |
| T3 Threshold | {T3_threshold} |
| T4 Threshold | {T4_threshold} |
| Veto (T0) | {veto_threshold} |
| Daily DD Stop | -5.0 R |

## Performance
| Metric | Value |
| :--- | :--- |
| **Total Trades** | {total_trades} |
| **Wins** | {wins} |
| **Losses** | {losses} |
| **Win Rate** | **{win_rate:.2%}** |
| **Total Return (R)** | **{total_R:+.2f} R** |
| **Expectancy / Trade** | **{total_R/total_trades:+.4f} R** if {total_trades} else N/A |
| **Ghost Trades (entry==exit time)** | {ghost_count} (should be 0) |
| **Observed Bricks** | {len(bricks_data)} |

{thresholds_table_md}
"""
    with open(SIM_OUT_DIR / "sim_report.md", "w") as f:
        f.write(report)

    logger.info("=========================================")
    logger.info(" CORRECTED SIMULATION COMPLETE")
    logger.info(f" Trades: {total_trades} | Wins: {wins} | Losses: {losses}")
    logger.info(f" Win Rate:      {win_rate:.2%}")
    logger.info(f" Total Return:  {total_R:+.2f} R")
    logger.info(f" Ghost Trades:  {ghost_count}")
    logger.info("=========================================")


if __name__ == "__main__":
    main()
