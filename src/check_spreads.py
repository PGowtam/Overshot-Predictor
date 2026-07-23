import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_parquet(BASE_DIR / "Data" / "blackbull_bricks_2026.parquet")

total_bricks = len(df)
high_spread_bricks = df[df['spread_current'] >= 1.0]

print(f"Total Bricks: {total_bricks}")
print(f"Bricks with spread >= 1.0: {len(high_spread_bricks)}")
print(f"Max spread: {df['spread_current'].max()}")
print(f"Mean spread: {df['spread_current'].mean()}")
