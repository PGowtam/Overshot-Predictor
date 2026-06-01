import os
import glob
import pandas as pd

# Only process H2 of 2023 (July - Dec) which matches the test set.
# Or process the entire year? Test set was 2023-07-03 to 2023-12-29.
files = glob.glob("Data/Raw/Ticks/2023/**/*.parquet", recursive=True)
files = [f for f in files if int(f.split('/')[-2]) >= 7]
print(f"Found {len(files)} parquet files for H2 2023.")

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
out_path = "data/xauusd_ticks_2023_H2.parquet"
combined.to_parquet(out_path)
print(f"Saved {len(combined)} ticks to {out_path}")
