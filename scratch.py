import pandas as pd
from pathlib import Path

tick_path = Path("Data/xauusd_ticks_2026.parquet")
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

print("TR stats:")
print(tr.describe())
print("\nATR stats:")
print(atr.describe())

print("\nSample daily bars (last 5):")
print(daily_bars.tail())
