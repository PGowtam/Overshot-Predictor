import pandas as pd
import numpy as np

print('Loading 2026 ticks parquet...')
df = pd.read_parquet('data/xauusd_ticks_2026.parquet')

print('\n--- Integrity Check ---')
print(f'Total Rows: {len(df):,}')
print(f'Null Values:\n{df.isnull().sum()}')

# Check monotonicity
is_monotonic = df['time_msc'].is_monotonic_increasing
print(f'\nTimestamps Strictly Monotonic Increasing: {is_monotonic}')

# Check time gaps
time_diffs = df['time_msc'].diff()
max_gap = time_diffs.max() / 1000 / 3600  # in hours
min_gap = time_diffs.min()
print(f'Max gap between ticks: {max_gap:.2f} hours (expected for weekends)')
print(f'Min gap between ticks: {min_gap} ms')

# Negative gaps?
neg_gaps = (time_diffs < 0).sum()
print(f'Negative time gaps (out of order): {neg_gaps}')

# Days calculation
df['date'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
unique_days = df['date'].nunique()
print(f'\nUnique Trading Days: {unique_days}')

d_min = df['date'].min()
d_max = df['date'].max()
print(f'Date Range: {d_min} to {d_max}')

print('\n--- Data Sample ---')
print(df.head(3))
