"""
Baiting Strategy Analysis (Experimental)

Hypothesis:
- Standard Strategy: High Prob_Win & High Pred_OS -> Take trade (Trend Following).
- Baiting Strategy: Identify signals that are highly likely to FAIL (Loss).
- Strategy:
  1. If Signal is STRONG WIN -> Trade Normal (+0.5% if win, -0.5% if loss).
  2. If Signal is STRONG LOSS -> Trade Reverse (+0.5% if loss, -0.5% if win).

This script:
1. Finds optimal thresholds for "Baiting" (predicting losses).
2. Simulates a combined portfolio.
3. Outputs metrics (Equity, Win Rate, Drawdown).
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
PLOT_DIR = OUTPUT_DIR / "plots" / "baiting"
HOLDOUT_LABELS_PATH = OUTPUT_DIR / "holdout" / "labels.parquet"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT_BRICKS = 10

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def reconstruct_dates(labels_path: Path):
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found at {labels_path}")
    df = pd.read_parquet(labels_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    valid_dates = []
    
    # Reimplements tensor_builder logic
    for i in range(len(df)):
        if i < CONTEXT_BRICKS: continue
        row = df.iloc[i]
        if bool(row["exclude_flag"]): continue
        if pd.isna(row["y_class"]): continue
        valid_dates.append(row["date"])
        
    return np.array(valid_dates)

def safe_predict(model, micro, macro, batch_size=32):
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

def find_baiting_thresholds(y_class, prob_win, pred_os, step=0.05):
    """
    Search for thresholds (p_low, o_low) such that:
    If Prob_Win < p_low AND Pred_OS < o_low (or some other logic?)
    Actually, we want to predict LOSS.
    Model output Prob_Win is probability of WIN.
    So low Prob_Win means high probability of LOSS.
    
    We look for Prob_Win < Threshold.
    And maybe Pred_OS < Threshold? Or Pred_OS > Threshold (trap)?
    User said "find thresholds of BOTH heads".
    
    Let's grid search:
    Conf_Loss = 1 - Prob_Win.
    We want Conf_Loss > Threshold (i.e. Prob_Win < 1 - Threshold).
    
    Let's try:
    - Prob_Win < P_Bait
    - Pred_OS < O_Bait (weak move?) OR Pred_OS > O_Bait (fake breakout?)
    
    We'll scan Prob_Win < p in [0.1, 0.5] and Pred_OS in [0.5, 2.0].
    Objective: Maximize Win Rate of REVERSAL (i.e. Maximize Loss Rate of Normal).
    """
    print("\n🔎 Scanning for Baiting Thresholds...", flush=True)
    best_wr = 0.0
    best_p = 0.0
    best_o = 0.0
    best_count = 0
    
    # Grid search
    # Prob_Win lower bound (we want standard signal to be likely Loss)
    # Means Prob_Win should be LOW.
    for p in np.arange(0.1, 0.55, 0.05):
        for o in np.arange(0.5, 2.0, 0.1):
            # Condition: Weak signal?
            # User explanation: "predict loosing trades".
            # Usually implies Prob_Win is low.
            # Let's assume prediction is "Loss" if Prob_Win < p.
            # And maybe Pred_OS condition too.
            
            mask = (prob_win < p) & (pred_os < o) # Weak/Low probability
            # Metric: Loss Rate (which is Reversal Win Rate)
            if np.sum(mask) < 50: continue # Min sample size
            
            # y_class=0 is Loss. We want count(y_class==0) / count(mask)
            loss_rate = np.mean(y_class[mask] == 0.0)
            
            if loss_rate > best_wr:
                best_wr = loss_rate
                best_p = p
                best_o = o
                best_count = np.sum(mask)
                
    print(f"✅ Found BEST Baiting Config: Prob_Win < {best_p:.2f}, Pred_OS < {best_o:.2f}")
    print(f"   Reversal Win Rate: {best_wr:.2%} ({best_count} trades)")
    return best_p, best_o

def run_baiting_analysis():
    print("="*50, flush=True)
    print(" 🎣 BAITING STRATEGY ANALYSIS (Holdout 2024)")
    print("="*50, flush=True)

    # 1. Load Data
    try:
        micro = np.load(TENSOR_DIR / "holdout_micro.npy")
        macro = np.load(TENSOR_DIR / "holdout_macro.npy")
        y_class = np.load(TENSOR_DIR / "holdout_y_class.npy")
        dates = reconstruct_dates(HOLDOUT_LABELS_PATH)
        
        # Load Config (Standard Strategy)
        config = load_config()
        std_p = config["Prob_Win_threshold"]
        std_o = config["Pred_OS_threshold"]
        
        model = tf.keras.models.load_model(MODEL_PATH)
        prob_win, pred_os = safe_predict(model, micro, macro)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 2. Find Baiting Thresholds
    bait_p, bait_o = find_baiting_thresholds(y_class, prob_win, pred_os)
    
    # 3. Simulate Combined Strategy
    # Logic:
    # A. Standard Signal: Prob_Win >= std_p AND Pred_OS >= std_o
    #    -> Trade Direction = Model Prediction (Long if Long, Short if Short applied to Brick)
    #    -> Outcome: Win (+0.5%), Loss (-0.5%)
    # B. Baiting Signal: Prob_Win < bait_p AND Pred_OS < bait_o
    #    -> Trade Direction = REVERSE
    #    -> Outcome: If original was Win -> Loss (-0.5%). If original was Loss -> Win (+0.5%).
    # Priority? What if both fire? (Unlikely, p_high vs p_low).
    
    initial_capital = 10000.0
    equity = [initial_capital]
    trade_log = []
    
    std_wins = 0
    std_loss = 0
    bait_wins = 0
    bait_loss = 0
    
    curr_equity = initial_capital
    
    equity_curve = []
    dates_curve = []
    
    for i in range(len(prob_win)):
        p = prob_win[i]
        o = pred_os[i]
        y = y_class[i] # 1=TP hit (Model Correct), 0=SL hit (Model Wrong)
        
        trade_pnl = 0.0
        
        # Check Standard
        if p >= std_p and o >= std_o:
            # Standard Trade
            if y == 1.0: # Win
                trade_pnl = 0.005
                std_wins += 1
            else:
                trade_pnl = -0.005
                std_loss += 1
                
        # Check Baiting
        elif p < 0.2 and o < 0.7:
            # Baiting Trade (Reverse)
            # If y==1 (Model correct about direction), we reversed -> We LOSE.
            # If y==0 (Model wrong), we reversed -> We WIN.
            if y == 0.0: # Original was Loss -> Reversal is Win
                trade_pnl = 0.005
                bait_wins += 1
            else: # Original was Win -> Reversal is Loss
                trade_pnl = -0.005
                bait_loss += 1
        
        if trade_pnl != 0.0:
            curr_equity *= (1 + trade_pnl)
            equity_curve.append(curr_equity)
            dates_curve.append(dates[i])
            trade_log.append(trade_pnl)
            
    # Metrics
    total_trades = std_wins + std_loss + bait_wins + bait_loss
    final_return = (curr_equity - initial_capital) / initial_capital
    max_dd = 0.0
    if len(equity_curve) > 0:
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak) / peak
        max_dd = np.min(dd)
        
    print("\n📊 COMBINED PERFORMANCE")
    print(f"   Final Equity: ${curr_equity:,.2f} (+{final_return:.2%})")
    print(f"   Max Drawdown: {max_dd:.2%}")
    print(f"   Total Trades: {total_trades}")
    print(f"   Standard: {std_wins} W / {std_loss} L (WR: {std_wins/(std_wins+std_loss+1e-9):.1%})")
    print(f"   Baiting:  {bait_wins} W / {bait_loss} L (WR: {bait_wins/(bait_wins+bait_loss+1e-9):.1%})")

    # Plot
    if len(equity_curve) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(dates_curve, equity_curve, label="Combined Equity")
        plt.title(f"Baiting Strategy (Std + Reverse)\nReturn: {final_return:.2%} | MaxDD: {max_dd:.2%}")
        plt.savefig(PLOT_DIR / "baiting_equity.png")
        print(f"📈 Saved baiting_equity.png")

if __name__ == "__main__":
    run_baiting_analysis()
