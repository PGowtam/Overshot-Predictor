import os
import json
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/gopo/Quant Projects/CAPSTONE/Overshot")
OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"
CV_DIR = OUTPUT_DIR / "exec" / "cv"

START_CAPITAL = 100.0
PCT_PER_TRADE = 0.005 # 0.5%
FIXED_PAYOUT = START_CAPITAL * PCT_PER_TRADE # $0.50

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

def calculate_return(count, win_rate):
    if count == 0:
        return 0.0, 0.0
    wins = count * win_rate
    losses = count - wins
    total_return = (wins * FIXED_PAYOUT) - (losses * FIXED_PAYOUT)
    return_per_trade = total_return / count
    return total_return, return_per_trade

def perform_grid_search(y_true, prob_win, pred_os, title="Standard Strategy"):
    print(f"\n📊 GRID SEARCH: {title}")
    results = []
    for p in np.arange(0.0, 1.1, 0.1):
        for o in np.arange(0.0, 4.1, 0.1):
            mask = (prob_win >= p) & (pred_os >= o)
            count = np.sum(mask)
            win_rate = np.mean(y_true[mask] == 1.0) if count > 0 else 0.0
            total_return, return_per_trade = calculate_return(count, win_rate)
            results.append({
                "Prob_Win": round(p, 1),
                "Pred_OS": round(o, 1),
                "Count": int(count),
                "WinRate": round(win_rate, 4),
                "Total_Return": round(total_return, 2),
                "Return_Per_Trade": round(return_per_trade, 4)
            })
    return pd.DataFrame(results)

def perform_baiting_grid_search(y_true, prob_win, pred_os, title="Baiting Strategy"):
    print(f"\n📊 GRID SEARCH: {title}")
    results = []
    for p in np.arange(0.0, 1.1, 0.1):
        for o in np.arange(0.0, 4.1, 0.1):
            mask = (prob_win < p) & (pred_os < o)
            count = np.sum(mask)
            reversal_wr = np.mean(y_true[mask] == 0.0) if count > 0 else 0.0
            total_return, return_per_trade = calculate_return(count, reversal_wr)
            results.append({
                "Prob_Win_LT": round(p, 1),
                "Pred_OS_LT": round(o, 1),
                "Count": int(count),
                "RevWinRate": round(reversal_wr, 4),
                "Total_Return": round(total_return, 2),
                "Return_Per_Trade": round(return_per_trade, 4)
            })
    return pd.DataFrame(results)

def run_optimization():
    print("="*50)
    print(" 🎯 MT5 BOT CONFIGURATION OPTIMIZATION (Return-Based)")
    print("="*50)

    # 1. Load data
    micro = np.load(TENSOR_DIR / "holdout_micro.npy")
    macro = np.load(TENSOR_DIR / "holdout_macro.npy")
    y_class = np.load(TENSOR_DIR / "holdout_y_class.npy")
    
    # 2. Extract predictions
    all_prob_wins = []
    all_pred_oss = []
    for fold in [1, 2, 3]:
        model = tf.keras.models.load_model(CV_DIR / f"fold_{fold}" / "model.keras")
        p, o = safe_predict(model, micro, macro)
        all_prob_wins.append(p)
        all_pred_oss.append(o)
    ens_prob_win = np.mean(all_prob_wins, axis=0)
    ens_pred_os = np.mean(all_pred_oss, axis=0)
    
    # 3. Standard
    df_std = perform_grid_search(y_class, ens_prob_win, ens_pred_os)
    avg_count_std = df_std[df_std['Count'] > 0]['Count'].mean()
    print(f"\nAverage Active Count (Standard): {avg_count_std:.2f}")
    
    # Selection: Highest WinRate provided Count >= Avg_Count
    qualified_std = df_std[df_std['Count'] >= avg_count_std]
    best_std = qualified_std.sort_values(by=['WinRate', 'Total_Return'], ascending=False).head(1)
    
    # 4. Baiting
    df_bait = perform_baiting_grid_search(y_class, ens_prob_win, ens_pred_os)
    avg_count_bait = df_bait[df_bait['Count'] > 0]['Count'].mean()
    print(f"Average Active Count (Baiting): {avg_count_bait:.2f}")
    
    # Selection: Highest RevWinRate provided Count >= Avg_Count
    qualified_bait = df_bait[df_bait['Count'] >= avg_count_bait]
    best_bait = qualified_bait.sort_values(by=['RevWinRate', 'Total_Return'], ascending=False).head(1)
    
    print("\n" + "#"*40)
    print("🏆 FINAL RECOMMENDED CONFIGURATION")
    print("#"*40)
    print(f"STANDARD: Prob_Win >= {best_std.iloc[0]['Prob_Win']}, Pred_OS >= {best_std.iloc[0]['Pred_OS']}")
    print(f" - Win Rate: {best_std.iloc[0]['WinRate']:.2%}")
    print(f" - Count: {best_std.iloc[0]['Count']}")
    print(f" - Total Return: ${best_std.iloc[0]['Total_Return']}")
    print(f" - Return/Trade: {best_std.iloc[0]['Return_Per_Trade']}")
    
    print("\n" + "-"*40)
    
    print(f"BAITING: Prob_Win < {best_bait.iloc[0]['Prob_Win_LT']}, Pred_OS < {best_bait.iloc[0]['Pred_OS_LT']}")
    print(f" - Rev Win Rate: {best_bait.iloc[0]['RevWinRate']:.2%}")
    print(f" - Count: {best_bait.iloc[0]['Count']}")
    print(f" - Total Return: ${best_bait.iloc[0]['Total_Return']}")
    print(f" - Return/Trade: {best_bait.iloc[0]['Return_Per_Trade']}")
    
    # Save detailed CSVs
    df_std.to_csv(OUTPUT_DIR / "std_grid_search_refined.csv", index=False)
    df_bait.to_csv(OUTPUT_DIR / "bait_grid_search_refined.csv", index=False)

if __name__ == "__main__":
    run_optimization()
