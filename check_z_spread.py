import pandas as pd
df = pd.read_parquet("/Users/gopo/Quant Projects/CAPSTONE/Overshot/data/xauusd_ticks_2026.parquet")
spread = df['ask'].astype(float) - df['bid'].astype(float)
print(f"Spread Mean: {spread.mean():.4f}")
print(f"Spread Std: {spread.std():.4f}")
