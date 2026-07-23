import src.rule_test_2026_v4_mp as mp_script
import pandas as pd
from datetime import date
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    df = pd.read_parquet("Data/xauusd_ticks_2026.parquet")
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    
    mp_script.global_daily_groups = {day: group for day, group in df.groupby('utc_day')}
    
    days = sorted(df['utc_day'].unique())
    target_day = days[10]
    print(f"Testing Day: {target_day}")
    
    log = mp_script.simulate_day(target_day)
    print(f"Returned {len(log)} trades.")
