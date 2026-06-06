import pandas as pd
import numpy as np
from pathlib import Path

# Load trade data
p = Path("outputs/trades_macro0_be0.parquet")
df = pd.read_parquet(p)

# Filter to only filled trades
df = df[df['fill_t_msc'].notnull()]
df['duration_min'] = df['duration_ms'] / (1000 * 60)

print(f"Total Filled Trades: {len(df)}")
wins = df[df['result'] == 'win']
losses = df[df['result'] == 'loss']

print(f"\n--- Trade Duration Stats (Minutes) ---")
print(f"Wins   - Median: {wins['duration_min'].median():.1f} | 75th: {wins['duration_min'].quantile(0.75):.1f} | 90th: {wins['duration_min'].quantile(0.90):.1f}")
print(f"Losses - Median: {losses['duration_min'].median():.1f} | 75th: {losses['duration_min'].quantile(0.75):.1f} | 90th: {losses['duration_min'].quantile(0.90):.1f}")

# Simulate different time limits
print("\n--- Timeout Simulation ---")
print("Timeout(min) | Trades | WR%   | Exp(R)")

# We simulate a force-close at market if duration > timeout
# Since we don't have the exact tick price at the timeout MS, we will assume a harsh
# penalty: A force close is a -0.5R loss on average (since it hasn't hit +2R or -1R).
# Even simpler: If we just count them as -1R to be safe, what happens to WR?
# Actually, the fairest way to test a timeout is to see how many of the *long-duration* trades ended up winning.
for timeout in [15, 30, 45, 60, 90, 120, 240, 480]:
    # Trades that resolved BEFORE timeout
    fast_trades = df[df['duration_min'] <= timeout]
    
    fast_wins = len(fast_trades[fast_trades['result'] == 'win'])
    fast_losses = len(fast_trades[fast_trades['result'] == 'loss'])
    fast_total = len(fast_trades)
    
    # Trades that exceeded timeout (would be force-closed)
    slow_trades = df[df['duration_min'] > timeout]
    slow_total = len(slow_trades)
    
    # What was the actual result of the slow trades?
    slow_wins = len(slow_trades[slow_trades['result'] == 'win'])
    slow_losses = len(slow_trades[slow_trades['result'] == 'loss'])
    
    if slow_total > 0:
        slow_wr = slow_wins / slow_total * 100
        print(f"If trade takes > {timeout}m, its eventual WR is {slow_wr:.1f}% ({slow_wins}/{slow_total})")

