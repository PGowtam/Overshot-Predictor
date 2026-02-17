
import pandas as pd
import numpy as np
import os
import sys
import glob
from datetime import datetime, timedelta, time
import dukascopy_python as dc
from numba import jit
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration
SYMBOL = "XAU/USD"
TEMP_DIR = "temp_ticks_outcomes"
INPUT_FILE = "Data/Processed/XAUUSDRENKO20-24_Tick_Optimized.csv"
OUTPUT_FILE = "Data/Processed/Outcomes/renko_with_tick_outcomes_no_be_XAUUSD20-24.csv"

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ----------------------------------------------------
# 1. Tick Downloading & Loading (Cached)
# ----------------------------------------------------

def fetch_day_ticks_cached(day_date):
    """
    Downloads tick data for a specific day and caches it to CSV.
    Returns DataFrame with columns: [timestamp, bid_price, ask_price]
    """
    # File name based on date
    cache_file = os.path.join(TEMP_DIR, f"{day_date.strftime('%Y-%m-%d')}.csv")
    
    # Check cache
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            # print(f"  [Cache Error] {e}, re-downloading...")
            os.remove(cache_file)

    # Download if not in cache
    try:
        import dukascopy_python as dc # Import locally for multiprocessing safety
        
        # Create day range (00:00 to 23:59:59.999)
        start_t = datetime.combine(day_date, time(0, 0, 0))
        end_t = datetime.combine(day_date, time(23, 59, 59, 999999))
        
        # print(f"  Downloading ticks for {day_date}...")
        # Use 'fetch' with exact arguments from help() output:
        # fetch(instrument, interval, offer_side, start, end, ...)
        
        ticks = dc.fetch(
            instrument=SYMBOL, 
            interval=dc.INTERVAL_TICK, 
            offer_side=dc.OFFER_SIDE_BID,
            start=start_t, 
            end=end_t
        )
        
        if ticks is None or len(ticks) == 0:
            return None
            
        # Format: timestamp, bid_price, ask_price
        df = pd.DataFrame(ticks)
        
        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        
        # If timestamp is in index, reset it
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp'})
            
        # Rename price columns if needed
        # Observed: 'bidprice', 'askprice'
        if 'bidprice' in df.columns:
            df = df.rename(columns={'bidprice': 'bid', 'askprice': 'ask'})
            
        # Keep relevant
        if 'bid' not in df.columns or 'timestamp' not in df.columns:
             print(f"  [Data Error] Unexpected columns: {df.columns.tolist()}")
             return None

        df = df[['timestamp', 'bid', 'ask']] 
        
        # Save cache
        df.to_csv(cache_file, index=False)
        
        return df
        
    except Exception as e:
        print(f"  [Download Error] {day_date}: {e}")
        return None

# ----------------------------------------------------
# 2. Strict Tick Simulation Logic (Numba) - NO BE
# ----------------------------------------------------

@jit(nopython=True)
def simulate_outcome_ticks_jit(
    tick_prices, 
    tick_timestamps_float, 
    entry_price, 
    brick_size, 
    is_buy, 
    start_ts_float, 
    end_ts_float
):
    """
    Simulates trade outcome by iterating ticks strictly.
    Returns: 1 (WIN), -1 (LOSS), -99 (OPEN/Unknown)
    NO BE Logic: Only TP or SL matters.
    """
    
    # Targets
    if is_buy:
        tp = entry_price + brick_size
        sl = entry_price - brick_size
    else: # Sell
        tp = entry_price - brick_size
        sl = entry_price + brick_size
        
    n = len(tick_prices)
    for i in range(n):
        ts = tick_timestamps_float[i]
        price = tick_prices[i]
        
        # Strict Time Filter: start_time < t <= end_time
        if ts <= start_ts_float:
            continue
        if ts > end_ts_float: # Reached next brick time
            break
            
        # Check Outcome
        if is_buy:
            # SL Check first? In same tick, we usually check High/Low. 
            # But line-by-line tick data gives price sequence.
            # So just check current price.
            
            if price <= sl:
                return -1
            if price >= tp:
                return 1
                
        else: # Sell
            if price >= sl:
                return -1
            if price <= tp:
                return 1
                
    return -99 # Still Open

# ----------------------------------------------------
# 3. Worker Function
# ----------------------------------------------------

def process_day(day_data_tuple):
    """
    Process outcomes for a single day's bricks.
    day_data_tuple: (day_date, list_of_brick_dicts_for_day, next_day_first_brick_date_if_any)
    """
    day_date, bricks, next_limit_date = day_data_tuple
    
    if not bricks:
        return []
        
    start_limit = bricks[0][1]['date']
    end_limit = bricks[-1][2]['date'] # Date of the next brick after the last one
    
    needed_days = pd.date_range(start=start_limit.date(), end=end_limit.date(), freq='D')
    
    full_ticks_dfs = []
    
    for d in needed_days:
        # fetch_day... handles empty.
        df = fetch_day_ticks_cached(d.date())
        if df is not None:
            full_ticks_dfs.append(df)
            
    if not full_ticks_dfs:
        return [] # No data
        
    # Merge and sort
    ticks_df = pd.concat(full_ticks_dfs).sort_values('timestamp').reset_index(drop=True)
    
    # Prepare Arrays for JIT
    # Use 'bid' price for everything (Renko construction standard)
    tick_prices = ticks_df['bid'].values
    tick_times = ticks_df['timestamp'].values.astype(np.float64) # Float seconds/ns
    
    results = []
    
    for idx, brick, next_brick in bricks:
        start_time = brick['date']
        end_time = next_brick['date']
        
        entry_price = brick['close']
        brick_size = brick['brick_size']
        uptrend = brick['uptrend']
        
        # Ensure uptrend is correct type
        if isinstance(uptrend, str):
            is_buy = uptrend.lower() == 'true'
        else:
            is_buy = bool(uptrend)
            
        start_ts_float = float(start_time.value)
        end_ts_float = float(end_time.value)
        
        # 1. Instant Check (Gap)
        if start_time == end_time:
            # Check next brick trend
            if isinstance(next_brick['uptrend'], str):
                next_is_buy = next_brick['uptrend'].lower() == 'true'
            else:
                next_is_buy = bool(next_brick['uptrend'])
                
            code = 1 if is_buy == next_is_buy else -1
            
        else:
            # 2. Simulate
            code = simulate_outcome_ticks_jit(
                tick_prices,
                tick_times,
                entry_price,
                brick_size,
                is_buy,
                start_ts_float,
                end_ts_float
            )
            
        # Map Code
        if code == 1:
            outcome = 'WIN'
        elif code == -1:
            outcome = 'LOSS'
        else:
            outcome = 'OPEN' # Should not happen if data is continuous
            
        results.append({
            'original_index': idx,
            'outcome': outcome
        })
        
    return results

# ----------------------------------------------------
# 4. Main
# ----------------------------------------------------

def main():
    print("Loading Renko Data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Dates
    # Use ISO8601 format to handle timestamps with/without microseconds robustly
    try:
        df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True).dt.tz_convert(None)
    except Exception:
        # Fallback to mixed if strict ISO fails
        df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True).dt.tz_convert(None)
    
    # Sort
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} bricks.")
    
    # Prepare Tasks
    # We prefer to chunk by DAY to optimize tick downloads.
    # Group bricks by date.
    df['day_group'] = df['date'].dt.date
    
    tasks = []
    
    unique_days = df['day_group'].unique()
    
    day_indices_map = df.groupby('day_group').indices # Dict: date -> integer index array
    
    total_bricks = len(df)
    
    for day in unique_days:
        indices = day_indices_map[day]
        
        # Build list of (i, brick_i, brick_i+1)
        day_task_items = []
        
        for i in indices:
            if i >= total_bricks - 1:
                continue # Skip last brick of dataset
                
            brick_curr = df.iloc[i]
            brick_next = df.iloc[i+1] # This works even if i+1 is in next day
            
            day_task_items.append((i, brick_curr, brick_next))
            
        if day_task_items:
            tasks.append((day, day_task_items, None))
            
    print(f"Prepared {len(tasks)} day tasks.")
    print("Starting Processing (Multiprocessing - NO BE)...")
    
    all_outcomes = []
    
    # Parallel Execution
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_day, t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                res = future.result()
                all_outcomes.extend(res)
            except Exception as e:
                print(f"Task Failed: {e}")
                
    # Merge Results
    print("Merging outcomes...")
    outcome_map = {item['original_index']: item['outcome'] for item in all_outcomes}
    
    # Apply to DF
    df['outcome'] = df.index.map(outcome_map)
    
    # Drop rows with no outcome
    full_len = len(df)
    df = df.dropna(subset=['outcome'])
    print(f"Dropped {full_len - len(df)} bricks (End of data/errors).")
    
    # Remove 'best_profit', 'day_group'
    cols_to_drop = ['best_profit', 'day_group']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")
    print(df['outcome'].value_counts())
    
    # Safe Cleanup
    try:
        print("Cleaning up temp ticks...")
        if os.path.exists(TEMP_DIR):
            for f in glob.glob(os.path.join(TEMP_DIR, "*.csv")):
                os.remove(f)
            os.rmdir(TEMP_DIR)
        print("Cleanup complete.")
    except:
        print("Cleanup partial/failed.")

if __name__ == "__main__":
    main()
