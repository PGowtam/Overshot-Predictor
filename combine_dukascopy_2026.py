import os
import glob
import pandas as pd

files = glob.glob("Data/Raw/Ticks/2026/**/*.parquet", recursive=True)
print(f"Found {len(files)} parquet files for Dukascopy 2026.")

dfs = []
for f in files:
    try:
        df = pd.read_parquet(f)
        # Rename timestamp to time_msc
        df['time_msc'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
        # Drop old timestamp
        df = df.drop(columns=['timestamp'])
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("Combining...")
combined = pd.concat(dfs, ignore_index=True)
print("Sorting...")
combined = combined.sort_values("time_msc").reset_index(drop=True)
out_path = "data/xauusd_ticks_dukascopy_2026.parquet"
combined.to_parquet(out_path)
print(f"Saved {len(combined)} ticks to {out_path}")
