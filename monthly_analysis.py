import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent

EXEC_DIR = BASE_DIR / "outputs" / "exec"
CV_DIR = EXEC_DIR / "cv"
HOLDOUT_DIR = EXEC_DIR / "holdout" / "tensors"

def safe_predict(model, micro, macro, batch_size=64):
    n = len(micro)
    prob_wins, pred_oss = [], []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        preds = model([micro[i:end], macro[i:end]], training=False)
        prob_wins.append(preds[0].numpy().flatten())
        pred_oss.append(preds[1].numpy().flatten())
    return np.concatenate(prob_wins), np.concatenate(pred_oss)

def main():
    try:
        df_labels = pd.read_parquet(EXEC_DIR / "holdout" / "labels.parquet")
    except Exception as e:
        print("Error reading holdout labels")
        return
    
    # Needs to match tensor_builder logic exactly for indices
    df_labels["date"] = pd.to_datetime(df_labels["date"], utc=True)
    
    # 1. Must have 10 bricks of context (index >= 9)
    # But filtering by i < 10 in tensor builder happens at the iteration level
    # Actually, we can just recreate the exact loop
    
    n_bricks = len(df_labels)
    valid_indices = []
    
    holdout_start = pd.Timestamp("2024-01-01", tz="UTC")
    
    for i in range(n_bricks):
        if i < 10:
            continue
        row = df_labels.iloc[i]
        if bool(row["exclude_flag"]):
            continue
        if pd.isna(row["y_class"]):
            continue
        if row["date"] >= holdout_start:
            valid_indices.append(i)

    df_holdout = df_labels.iloc[valid_indices].copy()
    
    h_micro = np.load(HOLDOUT_DIR / "holdout_micro.npy")
    h_macro = np.load(HOLDOUT_DIR / "holdout_macro.npy")
    h_yc = np.load(HOLDOUT_DIR / "holdout_y_class.npy")

    if len(df_holdout) != len(h_yc):
        print(f"Mismatch: DF holdout {len(df_holdout)} vs Tensor {len(h_yc)}.")
        return

    orig_model_path = BASE_DIR / "outputs" / "model.keras"
    orig_config_path = BASE_DIR / "outputs" / "config.json"
    
    if not orig_model_path.exists():
        print(f"Model not found at {orig_model_path}")
        return
        
    model = tf.keras.models.load_model(orig_model_path)
    with open(orig_config_path) as f:
        config = json.load(f)
        
    th_p = config["Prob_Win_threshold"]
    th_o = config["Pred_OS_threshold"]
    
    prob_win, pred_os = safe_predict(model, h_micro, h_macro)
    mask = (prob_win >= th_p) & (pred_os >= th_o)

    df_holdout['exec_model_signal'] = mask
    df_holdout['y_class_tensor'] = h_yc
    
    # Filter to only trades taken
    df_trades = df_holdout[df_holdout['exec_model_signal'] == True].copy()
    
    df_trades['month'] = pd.to_datetime(df_trades['date']).dt.to_period('M')
    
    # +1 for Win, -1 for Loss
    df_trades['score'] = df_trades['y_class_tensor'].apply(lambda x: 1 if x == 1 else -1)
    
    grouped = df_trades.groupby('month').agg(
        trades=('score', 'count'),
        wins=('y_class_tensor', 'sum'),
        score=('score', 'sum')
    )
    
    grouped['win_rate'] = grouped['wins'] / grouped['trades']
    
    print("\n--- 2024 Monthly Breakdown (Phase 9 Single Exec Model) ---")
    print(grouped)
    print("-" * 60)
    print(f"Total Trades: {grouped['trades'].sum()}")
    print(f"Total Wins:   {grouped['wins'].sum()}")
    print(f"Total Score:  {grouped['score'].sum()}")
    print(f"Net Win Rate: {grouped['wins'].sum() / grouped['trades'].sum():.2%}")

if __name__ == "__main__":
    main()
