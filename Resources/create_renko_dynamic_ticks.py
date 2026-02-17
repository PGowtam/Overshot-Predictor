import pandas as pd
import numpy as np
import os
import sys
import glob
from datetime import datetime, timedelta, time
import dukascopy_python as dc
from numba import jit
import warnings

warnings.filterwarnings('ignore')

# Configuration
SYMBOL = "XAU/USD"
TEMP_DIR = "temp_ticks"
OUTPUT_FILE = "Data/Processed/XAUUSDRENKO20-24_Tick_Optimized.csv"

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
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
    """
    Generates bricks from ticks.
    Timestamps are in nanoseconds (int64).
    """
    n = len(timestamps)
    current_brick_price = start_price
    
    # Dynamic Lists for Numba
    out_dates = []
    out_opens = []
    out_closes = []
    out_uptrends = []
    out_highs = []
    out_lows = []
    out_volumes = []
    
    uptrend = 0 # 0: None, 1: True, -1: False
    
    # Iterate through TICKS
    for i in range(start_index, n):
        ts = timestamps[i]
        price = prices[i]
        
        # Optimization: Skip if before window (should be handled by slice, but safe check)
        # if ts < window_start_ts: continue 
        
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
    """
    Simulates trades based on generated bricks against raw tick data.
    """
    n_bricks = len(b_dates)
    if n_bricks == 0:
        return 0.0
    
    daily_pnl = 0.0
    last_tick_idx = 0
    max_tick_idx = len(tick_ts)
    
    for i in range(n_bricks):
        entry_price = b_closes[i]
        uptrend = b_uptrends[i]
        start_time = b_dates[i]
        
        # Trade Params
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
            
        # End time determined by next brick
        if i < n_bricks - 1:
            end_time = b_dates[i+1] # Exclusive
            
            # Match "Optimized T6" Logic for Same-Timestamp Bricks (Gaps)
            if start_time == end_time:
                next_trend = b_uptrends[i+1]
                if next_trend == uptrend:
                    daily_pnl += 1.0 # Gap Continuation -> Instant Win (Weighted 1.0 in original)
                else:
                    daily_pnl -= 1.0 # Gap Reversal -> Instant Loss
                continue
        else:
            end_time = tick_ts[-1] # End of data
            
        # Fast forward ticks to start_time
        curr_idx = last_tick_idx
        while curr_idx < max_tick_idx and tick_ts[curr_idx] <= start_time:
            curr_idx += 1
            
        outcome = 0
        sl_moved_to_be = False
        
        scan_idx = curr_idx
        while scan_idx < max_tick_idx:
            ts = tick_ts[scan_idx]
            if ts >= end_time:
                break
                
            price = tick_prices[scan_idx]
            
            if trade_type == 1: # BUY
                current_sl = entry_price if sl_moved_to_be else sl_price
                if price <= current_sl:
                    outcome = -1 if not sl_moved_to_be else 0
                    break
                if price >= tp_price:
                    outcome = 1
                    break
                if price >= be_trigger:
                    sl_moved_to_be = True
            else: # SELL
                current_sl = entry_price if sl_moved_to_be else sl_price
                if price >= current_sl:
                    outcome = -1 if not sl_moved_to_be else 0
                    break
                if price <= tp_price:
                    outcome = 1
                    break
                if price <= be_trigger:
                    sl_moved_to_be = True
            
            scan_idx += 1
                        
        last_tick_idx = scan_idx 
        
        if outcome == 1:
            daily_pnl += 0.5
        elif outcome == -1:
            daily_pnl -= 0.5
            
    return daily_pnl


# ==========================================
# Data Management
# ==========================================

def fetch_day_ticks_cached(day_date):
    """
    Fetches ticks for a single day. Returns DataFrame.
    Uses TEMP_DIR caching.
    """
    file_path = os.path.join(TEMP_DIR, f"{day_date}.csv")
    
    if os.path.exists(file_path):
        try:
            # Explicit date format to avoid ambiguity
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=['timestamp'], date_format='ISO8601')
            
            # Explicitly convert index if not DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True, format='ISO8601')
            
            # Now safe to check tz (or it's already UTC from to_datetime(utc=True))
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
                
            return df
        except Exception as e:
            print(f"  Error reading cache {day_date}: {e}")
            try:
                os.remove(file_path) # Corrupt?
            except:
                pass
        
    # Download
    print(f"  Downloading ticks for {day_date}...")
    start_dt = datetime.combine(day_date, time.min)
    end_dt = datetime.combine(day_date, time.max)
    
    try:
        df = dc.fetch(
            instrument=SYMBOL,
            interval=dc.INTERVAL_TICK, # Raw Ticks
            offer_side=dc.OFFER_SIDE_BID,
            start=start_dt,
            end=end_dt
        )
        if not df.empty:
            # Ensure index is datetime (should be from library)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            
            df.to_csv(file_path)
            return df
    except Exception as e:
        print(f"  Error downloading {day_date}: {e}")
    
    return pd.DataFrame() # Empty if failed

def clean_cache(target_day, lookback_days=7):
    """Deletes files older than the lookback window."""
    keep_days = [target_day - timedelta(days=i) for i in range(lookback_days + 2)]
    keep_files = {f"{d}.csv" for d in keep_days}
    
    for f in os.listdir(TEMP_DIR):
        if f not in keep_files and f.endswith(".csv"):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
                print(f"  [Cache] Deleted old file: {f}")
            except:
                pass

# ==========================================
# Main Process
# ==========================================

from concurrent.futures import ProcessPoolExecutor, as_completed

def process_day(target_day):
    # Re-import inside process for Windows/start_method safety (optional but good practice)
    # Global constants (TEMP_DIR, SYMBOL) are available if defined at module level
    
    # print(f"Processing Target: {target_day}") # Avoid print in MP to prevent garbled output
    
    # 1. Find Valid Anchor Day (Dynamic Lookback)
    # Start looking from T-6 backwards
    anchor_day_date = None
    anchor_df = None
    
    initial_lookback = 6
    max_lookback = 20
    
    for i in range(initial_lookback, max_lookback + 1):
        candidate_date = target_day - timedelta(days=i)
        
        # Skip weekends to save API calls/time
        if candidate_date.weekday() >= 5: 
            continue
            
        try:
            df = fetch_day_ticks_cached(candidate_date)
            if not df.empty:
                anchor_day_date = candidate_date
                anchor_df = df
                break
        except:
            continue
            
    if anchor_day_date is None:
        return None # Silent failure or log to file
        
    # 2. Define Window based on valid Anchor
    # Use all days from anchor to target
    days_window = []
    curr = anchor_day_date
    while curr <= target_day:
        days_window.append(curr)
        curr += timedelta(days=1)
    
    # 3. Fetch Data for rest of window
    dfs = []
    if anchor_df is not None:
        dfs.append(anchor_df)
        
    # Fetch rest (skipping anchor which is already first)
    for d in days_window[1:]:
        df = fetch_day_ticks_cached(d)
        if not df.empty:
            dfs.append(df)
            
    if not dfs:
        return None
        
    full_df = pd.concat(dfs)
    full_df.sort_index(inplace=True)
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    
    # 3. Prepare Arrays for Numba
    if full_df.index.tz is None:
        full_df.index = full_df.index.tz_localize('UTC')
    else:
        full_df.index = full_df.index.tz_convert('UTC')
        
    timestamps = full_df.index.values.astype(np.int64)
    prices = full_df['bidPrice'].values
    
    # Anchor Logic
    anchor_day_date = days_window[0]
    target_day_ts = pd.Timestamp(target_day).tz_localize('UTC')
    t_minus_5_ts = pd.Timestamp(days_window[1]).tz_localize('UTC')
    
    if full_df.index.tz is None:
        full_df.index = full_df.index.tz_localize('UTC') 
        
    anchor_mask = (full_df.index.date == anchor_day_date)
    if not anchor_mask.any():
        return None
        
    anchor_high = prices[anchor_mask].max()
    anchor_low = prices[anchor_mask].min()
    anchor_open = prices[anchor_mask][0]
    
    # Target Open for Brick Size
    target_mask = (full_df.index.date == target_day)
    if not target_mask.any():
        target_open = prices[-1] 
    else:
        target_open = prices[target_mask][0]
        
    brick_size = (target_open * 0.00236)/2
    step_size = anchor_open * 0.00236 * 0.01
    
    candidates = np.arange(anchor_low, anchor_high + step_size/1000, step_size)[::-1]
    
    # Slice for Simulation (T-5 to Start of Target)
    sim_mask = (timestamps >= t_minus_5_ts.value) & (timestamps < target_day_ts.value)
    sim_ts = timestamps[sim_mask]
    sim_prices = prices[sim_mask]
    
    if len(sim_ts) == 0:
        return None

    # 4. Optimize
    best_profit = -float('inf')
    best_cand = candidates[0]
    
    anchor_prices = prices[anchor_mask]
    t5_start_int = t_minus_5_ts.value
    
    # Optimization Loop
    for cand in candidates:
        if not ((anchor_prices >= cand).any() and (anchor_prices <= cand).any()):
            continue
            
        if not (anchor_high >= cand >= anchor_low):
            continue
            
        # Find start idx based on logic
        anchor_indices = np.where(anchor_mask)[0]
        if len(anchor_indices) == 0: continue
            
        search_slice = prices[anchor_indices]
        idx_offset = np.argmax(np.abs(search_slice - cand) < brick_size)
        hit_idx = anchor_indices[idx_offset]
        
        # JIT Call
        b_dates, _, b_closes, b_uptrends, _, _, _ = generate_bricks_ticks_jit(
            timestamps, prices,
            hit_idx, cand, brick_size,
            t5_start_int
        )
        
        if not b_dates: continue
        
        # Filter History bricks (before Target Day)
        bd = np.array(b_dates)
        bc = np.array(b_closes)
        bu = np.array(b_uptrends)
        
        hist_mask = (bd < target_day_ts.value)
        
        pnl = simulate_profit_ticks_jit(
            bd[hist_mask], bc[hist_mask], bu[hist_mask],
            sim_ts, sim_prices,
            brick_size
        )
        
        if pnl > best_profit:
            best_profit = pnl
            best_cand = cand

    # 5. Generate Final (Target Day)
    anchor_indices = np.where(anchor_mask)[0]
    search_slice = prices[anchor_indices]
    idx_offset = np.argmax(np.abs(search_slice - best_cand) < brick_size)
    best_hit_idx = anchor_indices[idx_offset]

    b_dates, b_opens, b_closes, b_uptrends, b_highs, b_lows, b_volumes = generate_bricks_ticks_jit(
        timestamps, prices,
        best_hit_idx, best_cand, brick_size,
        t5_start_int
    )
    
    final_data = []
    
    target_start_val = target_day_ts.value
    target_end_val = target_start_val + np.timedelta64(1, 'D').astype('timedelta64[ns]').astype('int64')
    
    for i in range(len(b_dates)):
        ts_val = b_dates[i]
        
        if target_start_val <= ts_val < target_end_val:
            ts_dt = pd.Timestamp(ts_val, tz='UTC')
            
            start_seq = max(0, i-50)
            seq_snippet = b_uptrends[start_seq:i]
            seq_chars = ['1' if x == 1 else '0' for x in seq_snippet]
            seq_str = "".join(seq_chars)
            
            final_data.append({
                'date': ts_dt,
                'open': b_opens[i],
                'high': b_highs[i],
                'low': b_lows[i],
                'close': b_closes[i],
                'volume': 1.0, 
                'uptrend': (b_uptrends[i] == 1),
                'brick_size': brick_size,
                'sequence': seq_str,
                'best_profit': best_profit
            })
            
    # Safe Cache Cleanup (Parallel Friendly)
    # We can safely delete files that are WAY older than any possible lookback.
    # Max lookback is 20 days.
    # So if we are at `target_day`, any file older than `target_day - 30 days` is garbage.
    # This allows each process to clean up "ancient" history without race conditions on valid data.
    
    cutoff_date = target_day - timedelta(days=30)
    cleanup_file = os.path.join(TEMP_DIR, f"{cutoff_date}.csv")
    if os.path.exists(cleanup_file):
        try:
            os.remove(cleanup_file)
            # print(f"  [Clean] Removed old file: {cutoff_date}")
        except:
            pass # Race condition or file locked? Ignore.
            
    # Clean Cache done in main/worker? 
    # Better to keep cache for subsequent days if they overlap. 
    # But clean_cache logic is day specific... 
    # We can leave cache cleaning to a separate maintenance task or only clean VERY old files.
    # Parallel processes sharing cache is fine for reading.
    # clean_cache(target_day) # Disabled for MP safety on shared files
    
    return (target_day, final_data)

def main():
    # Use tqdm for progress bar if available
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs): return x

    start_date = datetime(2020, 1, 1).date()
    end_date = datetime(2023, 12, 31).date() 
    
    # Generate list of days
    days_to_process = []
    current_day = start_date
    while current_day <= end_date:
        if current_day.weekday() < 5: # Skip weekends
            days_to_process.append(current_day)
        current_day += timedelta(days=1)
        
    print(f"Starting parallel processing for {len(days_to_process)} days...")
    print(f"Using {os.cpu_count()} cores.")
    
    # Setup Output
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    all_results = []
    
    # Parallel Execution
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_day = {executor.submit(process_day, day): day for day in days_to_process}
        
        for future in tqdm(as_completed(future_to_day), total=len(days_to_process), desc="Processing"):
            day = future_to_day[future]
            try:
                result = future.result()
                if result and result[1]: # result is (date, data_list)
                    all_results.append(result)
                    # print(f"  Finished {result[0]}: {len(result[1])} bricks")
            except Exception as e:
                print(f"  Day {day} generated an exception: {e}")
                
    # Sort results by date
    all_results.sort(key=lambda x: x[0])
    
    # Write to CSV
    print("Writing to CSV...")
    header = True
    with open(OUTPUT_FILE, 'w') as f:
        start_write = True
        for date, data in all_results:
            if not data: continue
            
            df = pd.DataFrame(data)
            df.to_csv(f, header=start_write, index=False)
            start_write = False
            
    # Final Cleanup
    print("Cleaning up cache...")
    # Clean up everything before the end date to be safe? 
    # Or just clean everything in temp_ticks?
    # Since we are done, we can delete all files in temp_ticks that we downloaded.
    # But clean_cache logic is "delete older than X".
    # Let's clean everything older than end_date?
    # Actually, proper cleanup is:
    # 1. Get all dates we processed.
    # 2. Delete their cache files.
    
    # Simple aggressive cleanup:
    try:
        if os.path.exists(TEMP_DIR):
            for f in os.listdir(TEMP_DIR):
                if f.endswith(".csv"):
                    try:
                        os.remove(os.path.join(TEMP_DIR, f))
                    except:
                        pass
        print("Cache cleaned.")
    except Exception as e:
        print(f"Cleanup error: {e}")
            
    print("Done.")

if __name__ == "__main__":
    # Windows support necessitates this
    main()
