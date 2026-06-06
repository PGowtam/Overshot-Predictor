import pandas as pd
from pathlib import Path
import json

tick_path = Path("Data/xauusd_ticks_2026.parquet")
if not tick_path.exists():
    tick_path = Path("Data/xauusd_ticks_5ers_2026.parquet")

df = pd.read_parquet(tick_path)
df['datetime'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)
df_dt = df.set_index('datetime')

daily_bars = df_dt['bid'].resample('1d').ohlc()
daily_bars.dropna(inplace=True)

prev_close = daily_bars['close'].shift(1)
tr1 = daily_bars['high'] - daily_bars['low']
tr2 = (daily_bars['high'] - prev_close).abs()
tr3 = (daily_bars['low'] - prev_close).abs()

tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr = tr.rolling(window=14).mean()

# For 0.8x multiplier
brick_sizes = atr * 0.8
avg_brick_size = brick_sizes.mean()
avg_1_r = avg_brick_size * 1.5

print(f"Average Brick Size (0.8x): {avg_brick_size:.2f} points")
print(f"Average 1R Risk (1.5x Brick): {avg_1_r:.2f} points")
print(f"Total Points for 1.08 R: {(avg_1_r * 1.08):.2f} points")

