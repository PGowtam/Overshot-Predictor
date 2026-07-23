import pandas as pd
from src.rule_test_2026_v4_mp import simulate_day, global_daily_groups
from datetime import date

df = pd.read_parquet("Data/xauusd_ticks_2026.parquet")
df = df.sort_values('time_msc').reset_index(drop=True)
df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
for day, group in df.groupby('utc_day'):
    global_daily_groups[day] = group

# Get the 10th day
days = sorted(df['utc_day'].unique())
target_day = days[10]
print(f"Testing day: {target_day}")

# We will modify simulate_day slightly inside memory just by running it
import src.rule_test_2026_v4_mp as mp_script
mp_script.global_daily_groups = global_daily_groups

# Run it
try:
    log = mp_script.simulate_day(target_day)
    print(f"Returned trades: {len(log)}")
except Exception as e:
    print(f"Error: {e}")

