"""
Phase 9: Backtest Analysis (FR-BK-01) - Robust

Backtests the strategy on Holdout data (2024).
Parameters:
- Profit: +0.5% per trade
- Loss: -0.5% per trade
- Initial Capital: 10,000

Generates:
- Equity Curve
- Monthly Returns Heatmap (Matplotlib only)
- Quant Metrics
"""

import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"
MODEL_PATH = OUTPUT_DIR / "model.keras"
CONFIG_PATH = OUTPUT_DIR / "config.json"
PLOT_DIR = OUTPUT_DIR / "plots" / "backtests"
HOLDOUT_LABELS_PATH = OUTPUT_DIR / "holdout" / "labels.parquet"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_BRICKS = 10

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def reconstruct_dates(labels_path: Path):
    """Reconstruct valid dates corresponding to the tensors."""
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found at {labels_path}")
    
    df = pd.read_parquet(labels_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    
    valid_dates = []
    
    for i in range(len(df)):
        if i < CONTEXT_BRICKS:
            continue
            
        row = df.iloc[i]
        if bool(row["exclude_flag"]):
            continue
        if pd.isna(row["y_class"]):
            continue
            
        valid_dates.append(row["date"])
        
    return np.array(valid_dates)

def safe_predict(model, micro, macro, batch_size=32):
    """Manual batch prediction."""
    n_samples = len(micro)
    prob_wins = []
    pred_oss = []
    
    print(f"🔮 Predicting {n_samples} samples...", flush=True)
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_micro = micro[i:end]
        batch_macro = macro[i:end]
        preds = model([batch_micro, batch_macro], training=False)
        prob_wins.append(preds[0].numpy().flatten())
        pred_oss.append(preds[1].numpy().flatten())
        
    return np.concatenate(prob_wins), np.concatenate(pred_oss)

def calculate_max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = np.min(drawdown)
    return max_dd

def plot_heatmap_matplotlib(heatmap_data, filename):
    """Plot monthly returns heatmap using only matplotlib."""
    if heatmap_data.empty:
        return

    # Pivot table: Index=Year, Columns=Month (1..12)
    # Ensure all months 1..12 exist in columns
    for m in range(1, 13):
        if m not in heatmap_data.columns:
            heatmap_data[m] = np.nan
    
    # Sort columns 1..12
    cols = sorted([c for c in heatmap_data.columns if isinstance(c, (int, float))])
    heatmap_data = heatmap_data[cols]
    
    years = sorted(heatmap_data.index.tolist(), reverse=True) # Newest top
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Extract data grid matching years/months
    data = []
    for y in years:
        row = []
        for m in range(1, 13):
            if m in heatmap_data.columns:
                val = heatmap_data.loc[y, m]
                row.append(val)
            else:
                row.append(np.nan)
        data.append(row)
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(10, max(2, len(years) * 0.8 + 1)))
    
    # Plot imshow
    # We want green for positive, red for negative.
    # Normalize center=0
    im = ax.imshow(data, cmap="RdYlGn", vmin=-0.1, vmax=0.1, aspect='auto') # Assuming monthly returns within +/- 10%
    
    # Ticks
    ax.set_xticks(np.arange(len(months)))
    ax.set_yticks(np.arange(len(years)))
    ax.set_xticklabels(months)
    ax.set_yticklabels(years)
    
    # Annotate
    for i in range(len(years)):
        for j in range(len(months)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "black" if abs(val) < 0.05 else "white"
                text = ax.text(j, i, f"{val:.1%}",
                               ha="center", va="center", color=text_color, fontsize=9)
                               
    ax.set_title("Monthly Returns")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_backtest():
    print("="*50, flush=True)
    print(" PHASE 9: BACKTEST (Holdout 2024)", flush=True)
    print("="*50, flush=True)

    # 1. Load Data
    try:
        micro = np.load(TENSOR_DIR / "holdout_micro.npy")
        macro = np.load(TENSOR_DIR / "holdout_macro.npy")
        y_class = np.load(TENSOR_DIR / "holdout_y_class.npy")
        dates = reconstruct_dates(HOLDOUT_LABELS_PATH)
    except Exception as e:
        print(f"❌ Failed to load data: {e}", flush=True)
        return

    if len(micro) != len(dates):
        print(f"❌ Length mismatch: Tensors={len(micro)}, Dates={len(dates)}", flush=True)
        return
    else:
        print(f"✅ Loaded {len(micro)} samples from {dates.min()} to {dates.max()}", flush=True)

    # 2. Config & Model
    config = load_config()
    th_prob = config["Prob_Win_threshold"]
    th_os = config["Pred_OS_threshold"]
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 3. Predict
    prob_win, pred_os = safe_predict(model, micro, macro)
    
    # 4. Generate Signals
    signals = (prob_win >= th_prob) & (pred_os >= th_os)
    n_trades = np.sum(signals)
    print(f"📊 Trades Taken: {n_trades} ({n_trades/len(signals):.1%} of samples)", flush=True)

    # 5. Simulate Equity
    initial_capital = 10000.0
    win_pct = 0.005  # +0.5%
    loss_pct = -0.005 # -0.5%
    
    equity_series = pd.Series(index=dates, data=float(initial_capital)).sort_index()
    # We set initial capital. But as continuous series?
    # Better: iterate.
    
    curr_equity = initial_capital
    trade_returns = []
    
    # Create equity series matching dates
    # But dates are irregular (brick closures).
    # We want Equity AFTER each potential trade.
    equity_values = []
    
    for i in range(len(signals)):
        if signals[i]:
            outcome = y_class[i]
            ret = win_pct if outcome == 1.0 else loss_pct
            curr_equity *= (1 + ret)
            trade_returns.append(ret)
        
        equity_values.append(curr_equity)
        
    equity_series = pd.Series(data=equity_values, index=dates)
    
    final_equity = curr_equity
    total_return = (final_equity - initial_capital) / initial_capital
    max_dd = calculate_max_drawdown(equity_series.values)
    
    print(f"💰 Final Equity: ${final_equity:,.2f} (+{total_return:.2%})", flush=True)
    
    # Metrics
    if n_trades > 0:
        win_rate = np.mean(np.array(trade_returns) > 0)
        # Average Trade Return
        avg_trade = np.mean(trade_returns)
        std_trade = np.std(trade_returns)
        sharpe = avg_trade / std_trade if std_trade > 0 else 0
        ann_sharpe = sharpe * np.sqrt(n_trades) # Approx
    else:
        win_rate = 0
        ann_sharpe = 0

    print(f"📉 Max Drawdown: {max_dd:.2%}", flush=True)
    print(f"📊 Win Rate: {win_rate:.2%}", flush=True)
    print(f"⚡ Sharpe (Trade-based Ann): {ann_sharpe:.2f}", flush=True)
    
    # A. Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_series.index, equity_series.values, label="Equity")
    plt.title(f"2024 Holdout Equity Curve\nTotal Return: {total_return:.2%} | MaxDD: {max_dd:.2%}")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOT_DIR / "equity_curve.png")
    print(f"📈 Saved equity_curve.png", flush=True)
    
    # B. Monthly Returns Heatmap
    monthly_equity = equity_series.resample('ME').last()
    
    # To get monthly return: (End_Month - End_PrevMonth) / End_PrevMonth
    # Need to prepend initial capital for first month calculation
    month_start = monthly_equity.index[0] - pd.DateOffset(months=1) # Approx
    # Actually just insert 10000 at start
    initial_series = pd.Series([initial_capital], index=[monthly_equity.index[0] - pd.DateOffset(days=1)])
    # Wait, simple pct_change working on cumulative equity is correct for month-over-month.
    # First value of pct_change is NaN. We fill it manually.
    
    pct_change = monthly_equity.pct_change()
    # First month return is (monthly_equity[0] - 10000) / 10000
    first_ret = (monthly_equity.iloc[0] - initial_capital) / initial_capital
    pct_change.iloc[0] = first_ret
    
    monthly_df = pd.DataFrame({
        'Year': pct_change.index.year,
        'Month': pct_change.index.month,
        'Return': pct_change.values
    })
    
    heatmap_data = monthly_df.pivot(index='Year', columns='Month', values='Return')
    plot_heatmap_matplotlib(heatmap_data, PLOT_DIR / "monthly_heatmap.png")
    print(f"📅 Saved monthly_heatmap.png", flush=True)

    # Save Metrics
    metrics = {
        "Initial Capital": initial_capital,
        "Final Equity": final_equity,
        "Total Return": total_return,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": ann_sharpe,
        "Win Rate": win_rate,
        "Inferred Monthly Returns": heatmap_data.to_dict()
    }
    with open(PLOT_DIR / "metrics.json", "w") as f:
        # Convert df to dict requires care with NaNs/types
        # Just dump summary metrics
        del metrics["Inferred Monthly Returns"]
        json.dump(metrics, f, indent=4)
        
    print("✅ Backtest Complete.", flush=True)

if __name__ == "__main__":
    run_backtest()
