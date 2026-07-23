import pandas as pd
df = pd.read_parquet("/Users/gopo/Quant Projects/CAPSTONE/Overshot/data/xauusd_ticks_2026.parquet")
print(df.columns)
print(df.head(2))
