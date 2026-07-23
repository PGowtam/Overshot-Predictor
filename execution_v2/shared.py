"""
Shared utilities for execution engineering analysis.
All functions use EXECUTION pricing (ask for long entry, bid for short entry).
"""
import numpy as np, pandas as pd, tensorflow as tf, json
from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent
TICK_DIR = BASE_DIR / "Data" / "Raw" / "Ticks"
OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_tick_cache = {}

def load_ticks_for_date(date):
    key = (date.year, date.month, date.day)
    if key in _tick_cache:
        return _tick_cache[key]
    p = TICK_DIR / str(date.year) / f"{date.month:02d}" / f"{date.day:02d}.parquet"
    if not p.exists():
        df = pd.DataFrame(columns=["timestamp","bid","bid_vol","ask","ask_vol"])
    else:
        df = pd.read_parquet(p)
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    if len(_tick_cache) > 40:
        oldest = sorted(_tick_cache.keys())[:20]
        for k in oldest: del _tick_cache[k]
    _tick_cache[key] = df
    return df

def load_ticks_after(brick_close, max_ticks=5000):
    """Load ticks after brick close, spanning multiple days."""
    cd = brick_close.normalize(); frames = []
    for offset in range(30):
        dd = cd + timedelta(days=offset)
        t = load_ticks_for_date(dd)
        if len(t) > 0:
            f = t[t["timestamp"] > brick_close]
            if len(f) > 0: frames.append(f)
            if sum(len(x) for x in frames) >= max_ticks: break
    if not frames:
        return pd.DataFrame(columns=["timestamp","bid","bid_vol","ask","ask_vol"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").head(max_ticks).reset_index(drop=True)

def resolve_trade_exec(entry_tick, brick_size, is_long, future_ticks, tp_mult=1.0, sl_mult=1.0):
    """Resolve trade with EXECUTION pricing and arbitrary TP/SL multipliers.
    
    Entry: ask (long) or bid (short)
    Exit scan: bid (long hits TP/SL) or ask (short hits TP/SL)
    
    Returns: (outcome, ticks_to_resolve, entry_price, exit_price)
      outcome: 1=WIN, 0=LOSS, -1=UNRESOLVED
    """
    if is_long:
        entry_price = float(entry_tick["ask"])
    else:
        entry_price = float(entry_tick["bid"])
    
    tp_dist = brick_size * tp_mult
    sl_dist = brick_size * sl_mult
    
    if is_long:
        tp_level = entry_price + tp_dist
        sl_level = entry_price - sl_dist
    else:
        tp_level = entry_price - tp_dist
        sl_level = entry_price + sl_dist
    
    bids = future_ticks["bid"].values
    asks = future_ticks["ask"].values
    
    for i in range(len(bids)):
        if is_long:
            exit_price = bids[i]  # Sell at bid
            if exit_price >= tp_level: return 1, i, entry_price, exit_price
            if exit_price <= sl_level: return 0, i, entry_price, exit_price
        else:
            exit_price = asks[i]  # Buy back at ask
            if exit_price <= tp_level: return 1, i, entry_price, exit_price
            if exit_price >= sl_level: return 0, i, entry_price, exit_price
    
    return -1, len(bids), entry_price, 0.0

def load_holdout_with_predictions():
    """Load holdout labels, tensors, and model predictions.
    Returns DataFrame with columns: date, brick_size, uptrend, y_class, y_mag, prob_win, pred_os
    """
    # Labels
    hlp = OUTPUT_DIR / "holdout" / "labels.parquet"
    if not hlp.exists(): hlp = OUTPUT_DIR / "labels.parquet"
    labels = pd.read_parquet(hlp)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    if "exclude_flag" in labels.columns:
        labels = labels[~labels["exclude_flag"]]
    if labels["date"].min().year < 2024:
        labels = labels[labels["date"] >= "2024-01-01"]
    labels = labels[labels["y_class"].notna()].reset_index(drop=True)
    
    # Model predictions
    model = tf.keras.models.load_model(OUTPUT_DIR / "model.keras")
    micro = np.load(TENSOR_DIR / "holdout_micro.npy")
    macro = np.load(TENSOR_DIR / "holdout_macro.npy")
    y_class = np.load(TENSOR_DIR / "holdout_y_class.npy")
    y_mag = np.load(TENSOR_DIR / "holdout_y_mag.npy")
    
    pw, po = [], []
    for i in range(0, len(micro), 64):
        e = min(i + 64, len(micro))
        p = model([micro[i:e], macro[i:e]], training=False)
        pw.append(p[0].numpy().flatten())
        po.append(p[1].numpy().flatten())
    pw, po = np.concatenate(pw), np.concatenate(po)
    
    # Align: labels and tensors may have different counts
    n = min(len(labels), len(pw))
    df = labels.iloc[:n].copy()
    df["prob_win"] = pw[:n]
    df["pred_os"] = po[:n]
    df["y_class_tensor"] = y_class[:n]
    df["y_mag_tensor"] = y_mag[:n]
    
    del model, micro, macro
    tf.keras.backend.clear_session()
    
    return df

def compute_expectancy(wr, tp_mult, sl_mult):
    """E = WR * TP - (1-WR) * SL"""
    return wr * tp_mult - (1 - wr) * sl_mult

def breakeven_wr(tp_mult, sl_mult):
    """WR needed for E=0"""
    return sl_mult / (tp_mult + sl_mult)
