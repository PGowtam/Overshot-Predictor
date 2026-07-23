import pandas as pd
import numpy as np
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent

def compute_metrics(trades_df, rr):
    if len(trades_df) == 0:
        return 0, 0, 0.0, 0.0
    wins = (trades_df['reversion_depth'] >= rr).sum()
    total = len(trades_df)
    losses = total - wins
    wr = wins / total if total > 0 else 0
    pf = (wins * rr) / (losses * 1.0 + 1e-8)
    pnl = (wins * rr) - losses
    return total, wins, wr, pf, pnl

def run_sweep():
    parquet_path = BASE_DIR / "outputs" / "sim_labels_v4" / "v4_percentiles_labels.parquet"
    df = pd.read_parquet(parquet_path)
    
    # Ensure sorted by time (do NOT dedup yet, percentiles need full history)
    df['time_pd'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Use actual pandas Date objects for day arithmetic
    df['utc_day'] = pd.to_datetime(df['utc_day']).dt.date
    
    spreads = df['spread_current'].values
    ofis = df['abs_ofi_peak'].values
    years = df['year'].values
    
    n_rows = len(df)
    
    day_windows = [1, 2, 3, 5, 7, 10, 15, 20, 30, 60, 100]
    brick_windows = [50, 100, 200, 500, 1000, 2000, 5000]
    
    results = []
    
    # 1. Day-Based Sweep (Exactly matches RegimeTrackerV4)
    print("Running Day-Based Sweep...")
    days = np.array(sorted(df['utc_day'].unique()))
    
    for d in day_windows:
        t0 = time.time()
        spread_pct = np.full(n_rows, np.nan)
        ofi_pct = np.full(n_rows, np.nan)
        
        for day in days:
            start_day = day - pd.Timedelta(days=d)
            # Find history indices
            hist_mask = (df['utc_day'] >= start_day) & (df['utc_day'] < day)
            curr_mask = (df['utc_day'] == day)
            
            if hist_mask.sum() >= 100:
                hist_spread = np.sort(spreads[hist_mask])
                hist_ofi = np.sort(ofis[hist_mask])
                
                curr_spreads = spreads[curr_mask]
                curr_ofis = ofis[curr_mask]
                
                sp_pct = (np.searchsorted(hist_spread, curr_spreads, side='right') / len(hist_spread)) * 100.0
                op_pct = (np.searchsorted(hist_ofi, curr_ofis, side='right') / len(hist_ofi)) * 100.0
                
                spread_pct[curr_mask] = sp_pct
                ofi_pct[curr_mask] = op_pct
                
        df['temp_sp_pct'] = spread_pct
        df['temp_op_pct'] = ofi_pct
        
        df_dedup = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        for period, y_mask in [("IS (21-25)", (df_dedup['year'] >= 2021) & (df_dedup['year'] <= 2025)), ("OOS (26)", df_dedup['year'] == 2026)]:
            mask = y_mask & (df_dedup['temp_sp_pct'] >= 99) & (df_dedup['temp_op_pct'] <= 3)
            trades = df_dedup[mask]
            
            for rr in [1.0, 1.5, 2.0]:
                total, wins, wr, pf, pnl = compute_metrics(trades, rr)
                results.append({
                    'Type': 'Day',
                    'Window': d,
                    'Period': period,
                    'RR': rr,
                    'Trades': total,
                    'WR': wr,
                    'PF': pf,
                    'PnL': pnl
                })
        print(f"  Completed {d} days in {time.time()-t0:.2f}s")

    # 2. Brick-Based Sweep
    print("\nRunning Brick-Based Sweep...")
    for b in brick_windows:
        t0 = time.time()
        spread_pct = np.full(n_rows, np.nan)
        ofi_pct = np.full(n_rows, np.nan)
        
        # For small arrays, a loop is fine. 40k iterations takes ~1-2s in python.
        for i in range(n_rows):
            start_idx = max(0, i - b)
            if i - start_idx >= min(100, b):
                hist_spread = np.sort(spreads[start_idx:i])
                hist_ofi = np.sort(ofis[start_idx:i])
                
                spread_pct[i] = (np.searchsorted(hist_spread, spreads[i], side='right') / len(hist_spread)) * 100.0
                ofi_pct[i] = (np.searchsorted(hist_ofi, ofis[i], side='right') / len(hist_ofi)) * 100.0
                
        df['temp_sp_pct'] = spread_pct
        df['temp_op_pct'] = ofi_pct
        
        df_dedup = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        for period, y_mask in [("IS (21-25)", (df_dedup['year'] >= 2021) & (df_dedup['year'] <= 2025)), ("OOS (26)", df_dedup['year'] == 2026)]:
            mask = y_mask & (df_dedup['temp_sp_pct'] >= 99) & (df_dedup['temp_op_pct'] <= 3)
            trades = df_dedup[mask]
            
            for rr in [1.0, 1.5, 2.0]:
                total, wins, wr, pf, pnl = compute_metrics(trades, rr)
                results.append({
                    'Type': 'Brick',
                    'Window': b,
                    'Period': period,
                    'RR': rr,
                    'Trades': total,
                    'WR': wr,
                    'PF': pf,
                    'PnL': pnl
                })
        print(f"  Completed {b} bricks in {time.time()-t0:.2f}s")
        
    res_df = pd.DataFrame(results)
    out_path = BASE_DIR / "outputs" / "lookback_sweep_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    
if __name__ == "__main__":
    run_sweep()
