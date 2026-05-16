import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta, time
from numba import jit
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# Configuration
SYMBOL = "XAU/USD"
OUTPUT_FILE = "Data/Processed/XAUUSD_Holdout_K00295.csv"
TICK_BASE_DIR = "Data/Raw/Ticks"

# Ensure directories exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ==========================================
# Core Renko Logic (Tick-Based + Numba)
# ==========================================

@jit(nopython=True)
def generate_bricks_ticks_jit(
    timestamps, prices, # Raw tick arrays (int64 ns, float64)
    start_index, start_price, brick_size, 
    window_start_ts # int64 timestamp
):
    n = len(timestamps)
    current_brick_price = start_price
    
    out_dates = []
    out_opens = []
    out_closes = []
    out_uptrends = []
    out_highs = []
    out_lows = []
    out_volumes = []
    
    uptrend = 0
    
    for i in range(start_index, n):
        ts = timestamps[i]
        price = prices[i]
        
        if uptrend == 0:
            if price >= current_brick_price + brick_size:
                while price >= current_brick_price + brick_size:
                    current_brick_price += brick_size
                    if ts >= window_start_ts:
                        out_dates.append(ts)
                        out_opens.append(current_brick_price - brick_size)
                        out_closes.append(current_brick_price)
                        out_uptrends.append(1)
                        out_highs.append(current_brick_price)
                        out_lows.append(current_brick_price - brick_size)
                        out_volumes.append(0.0)
                uptrend = 1
            elif price <= current_brick_price - brick_size:
                while price <= current_brick_price - brick_size:
                    current_brick_price -= brick_size
                    if ts >= window_start_ts:
                        out_dates.append(ts)
                        out_opens.append(current_brick_price + brick_size)
                        out_closes.append(current_brick_price)
                        out_uptrends.append(-1)
                        out_highs.append(current_brick_price + brick_size)
                        out_lows.append(current_brick_price)
                        out_volumes.append(0.0)
                uptrend = -1
        else:
            if uptrend == 1:
                if price >= current_brick_price + brick_size:
                    while price >= current_brick_price + brick_size:
                        current_brick_price += brick_size
                        if ts >= window_start_ts:
                            out_dates.append(ts)
                            out_opens.append(current_brick_price - brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(1)
                            out_highs.append(current_brick_price)
                            out_lows.append(current_brick_price - brick_size)
                            out_volumes.append(0.0)
                elif price <= current_brick_price - 2 * brick_size:
                    current_brick_price -= 2 * brick_size
                    if ts >= window_start_ts:
                        out_dates.append(ts)
                        out_opens.append(current_brick_price + brick_size)
                        out_closes.append(current_brick_price)
                        out_uptrends.append(-1)
                        out_highs.append(current_brick_price + brick_size)
                        out_lows.append(current_brick_price)
                        out_volumes.append(0.0)
                    uptrend = -1
                    while price <= current_brick_price - brick_size:
                        current_brick_price -= brick_size
                        if ts >= window_start_ts:
                            out_dates.append(ts)
                            out_opens.append(current_brick_price + brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(-1)
                            out_highs.append(current_brick_price + brick_size)
                            out_lows.append(current_brick_price)
                            out_volumes.append(0.0)
            else: # Downtrend
                if price <= current_brick_price - brick_size:
                    while price <= current_brick_price - brick_size:
                        current_brick_price -= brick_size
                        if ts >= window_start_ts:
                            out_dates.append(ts)
                            out_opens.append(current_brick_price + brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(-1)
                            out_highs.append(current_brick_price + brick_size)
                            out_lows.append(current_brick_price)
                            out_volumes.append(0.0)
                elif price >= current_brick_price + 2 * brick_size:
                    current_brick_price += 2 * brick_size
                    if ts >= window_start_ts:
                        out_dates.append(ts)
                        out_opens.append(current_brick_price - brick_size)
                        out_closes.append(current_brick_price)
                        out_uptrends.append(1)
                        out_highs.append(current_brick_price)
                        out_lows.append(current_brick_price - brick_size)
                        out_volumes.append(0.0)
                    uptrend = 1
                    while price >= current_brick_price + brick_size:
                        current_brick_price += brick_size
                        if ts >= window_start_ts:
                            out_dates.append(ts)
                            out_opens.append(current_brick_price - brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(1)
                            out_highs.append(current_brick_price)
                            out_lows.append(current_brick_price - brick_size)
                            out_volumes.append(0.0)

    return out_dates, out_opens, out_closes, out_uptrends, out_highs, out_lows, out_volumes

@jit(nopython=True)
def simulate_profit_ticks_jit(
    b_dates, b_closes, b_uptrends,
    tick_ts, tick_prices,
    brick_size
):
    n_bricks = len(b_dates)
    if n_bricks == 0: return 0.0
    
    daily_pnl = 0.0
    last_tick_idx = 0
    max_tick_idx = len(tick_ts)
    
    for i in range(n_bricks):
        entry_price = b_closes[i]
        uptrend = b_uptrends[i]
        start_time = b_dates[i]
        
        if uptrend == 1:
            tp_price = entry_price + brick_size
            sl_price = entry_price - brick_size
            be_trigger = entry_price + (0.3125 * brick_size)
            trade_type = 1
        else:
            tp_price = entry_price - brick_size
            sl_price = entry_price + brick_size
            be_trigger = entry_price - (0.3125 * brick_size)
            trade_type = -1
            
        if i < n_bricks - 1:
            end_time = b_dates[i+1]
            if start_time == end_time:
                next_trend = b_uptrends[i+1]
                if next_trend == uptrend: daily_pnl += 1.0
                else: daily_pnl -= 1.0
                continue
        else:
            end_time = tick_ts[-1]
            
        curr_idx = last_tick_idx
        while curr_idx < max_tick_idx and tick_ts[curr_idx] <= start_time:
            curr_idx += 1
            
        outcome = 0
        sl_moved_to_be = False
        scan_idx = curr_idx
        while scan_idx < max_tick_idx:
            ts = tick_ts[scan_idx]
            if ts >= end_time: break
            price = tick_prices[scan_idx]
            if trade_type == 1:
                current_sl = entry_price if sl_moved_to_be else sl_price
                if price <= current_sl:
                    outcome = -1 if not sl_moved_to_be else 0
                    break
                if price >= tp_price:
                    outcome = 1
                    break
                if price >= be_trigger: sl_moved_to_be = True
            else:
                current_sl = entry_price if sl_moved_to_be else sl_price
                if price >= current_sl:
                    outcome = -1 if not sl_moved_to_be else 0
                    break
                if price <= tp_price:
                    outcome = 1
                    break
                if price <= be_trigger: sl_moved_to_be = True
            scan_idx += 1
        last_tick_idx = scan_idx 
        if outcome == 1: daily_pnl += 0.5
        elif outcome == -1: daily_pnl -= 0.5
            
    return daily_pnl

# ==========================================
# Data Management (Local Parquet)
# ==========================================

def fetch_day_ticks_local(day_date):
    path = os.path.join(TICK_BASE_DIR, str(day_date.year), f"{day_date.month:02d}", f"{day_date.day:02d}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if 'bid' in df.columns:
            df.rename(columns={'bid': 'bidPrice', 'ask': 'askPrice', 'bid_vol': 'bidVolume', 'ask_vol': 'askVolume'}, inplace=True)
        if df.index.name != 'timestamp' and 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        return df
    return pd.DataFrame()

# ==========================================
# Main Process
# ==========================================

def process_day(target_day):
    anchor_day_date = None
    anchor_df = None
    
    for i in range(6, 21):
        candidate_date = target_day - timedelta(days=i)
        if candidate_date.weekday() >= 5: continue
        df = fetch_day_ticks_local(candidate_date)
        if not df.empty:
            anchor_day_date = candidate_date
            anchor_df = df
            break
            
    if anchor_day_date is None: return None
        
    days_window = []
    curr = anchor_day_date
    while curr <= target_day:
        days_window.append(curr)
        curr += timedelta(days=1)
    
    dfs = [anchor_df]
    for d in days_window[1:]:
        df = fetch_day_ticks_local(d)
        if not df.empty: dfs.append(df)
            
    if not dfs: return None
        
    full_df = pd.concat(dfs).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    
    timestamps = full_df.index.values.astype(np.int64)
    prices = full_df['bidPrice'].values
    
    anchor_mask = (full_df.index.date == anchor_day_date)
    if not anchor_mask.any(): return None
        
    anchor_high, anchor_low, anchor_open = prices[anchor_mask].max(), prices[anchor_mask].min(), prices[anchor_mask][0]
    
    target_mask = (full_df.index.date == target_day)
    target_open = prices[target_mask][0] if target_mask.any() else prices[-1]
        
    brick_size = target_open * 0.00295 # NEW MULTIPLIER
    step_size = anchor_open * 0.00236 * 0.01
    candidates = np.arange(anchor_low, anchor_high + step_size/1000, step_size)[::-1]
    
    target_day_ts = pd.Timestamp(target_day).tz_localize('UTC')
    t_minus_5_ts = pd.Timestamp(days_window[1]).tz_localize('UTC')
    sim_mask = (timestamps >= t_minus_5_ts.value) & (timestamps < target_day_ts.value)
    sim_ts, sim_prices = timestamps[sim_mask], prices[sim_mask]
    
    if len(sim_ts) == 0: return None

    best_profit, best_cand = -float('inf'), candidates[0]
    anchor_indices = np.where(anchor_mask)[0]
    anchor_prices = prices[anchor_mask]
    
    for cand in candidates:
        if not (anchor_high >= cand >= anchor_low): continue
        if not ((anchor_prices >= cand).any() and (anchor_prices <= cand).any()): continue
        
        search_slice = prices[anchor_indices]
        idx_offset = np.argmax(np.abs(search_slice - cand) < brick_size)
        hit_idx = anchor_indices[idx_offset]
        
        b_dates, _, b_closes, b_uptrends, _, _, _ = generate_bricks_ticks_jit(
            timestamps, prices, hit_idx, cand, brick_size, t_minus_5_ts.value
        )
        if not b_dates: continue
        
        bd, bc, bu = np.array(b_dates), np.array(b_closes), np.array(b_uptrends)
        hist_mask = (bd < target_day_ts.value)
        pnl = simulate_profit_ticks_jit(bd[hist_mask], bc[hist_mask], bu[hist_mask], sim_ts, sim_prices, brick_size)
        
        if pnl > best_profit:
            best_profit, best_cand = pnl, cand

    idx_offset = np.argmax(np.abs(prices[anchor_indices] - best_cand) < brick_size)
    best_hit_idx = anchor_indices[idx_offset]

    b_dates, b_opens, b_closes, b_uptrends, b_highs, b_lows, _ = generate_bricks_ticks_jit(
        timestamps, prices, best_hit_idx, best_cand, brick_size, t_minus_5_ts.value
    )
    
    final_data = []
    target_start_val = target_day_ts.value
    target_end_val = target_start_val + 86400000000000 # 1 day in ns
    
    for i in range(len(b_dates)):
        ts_val = b_dates[i]
        if target_start_val <= ts_val < target_end_val:
            seq_snippet = b_uptrends[max(0, i-50):i]
            seq_str = "".join(['1' if x == 1 else '0' for x in seq_snippet])
            final_data.append({
                'date': pd.Timestamp(ts_val, tz='UTC'),
                'open': b_opens[i], 'high': b_highs[i], 'low': b_lows[i], 'close': b_closes[i],
                'volume': 1.0, 'uptrend': (b_uptrends[i] == 1), 'brick_size': brick_size,
                'sequence': seq_str, 'best_profit': best_profit
            })
    return (target_day, final_data)

def main():
    try: from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs): return x

    # Holdout is 2024
    start_date, end_date = datetime(2024, 1, 1).date(), datetime(2024, 12, 31).date()
    days_to_process = [d for d in (start_date + timedelta(n) for n in range((end_date - start_date).days + 1)) if d.weekday() < 5]
        
    print(f"Rebuilding Renko Holdout (2024) with K=0.00295...")
    all_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_day = {executor.submit(process_day, day): day for day in days_to_process}
        for future in tqdm(as_completed(future_to_day), total=len(days_to_process), desc="Renko"):
            try:
                res = future.result()
                if res and res[1]: all_results.append(res)
            except Exception as e: print(f"Error: {e}")
                
    all_results.sort(key=lambda x: x[0])
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        first = True
        for _, data in all_results:
            df = pd.DataFrame(data)
            df.to_csv(f, header=first, index=False)
            first = False
    print("Done.")

if __name__ == "__main__": main()
