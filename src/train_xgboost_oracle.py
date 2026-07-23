import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(years, data_dir):
    dfs = []
    for y in years:
        f = data_dir / f"v4_features_{y}.parquet"
        if f.exists():
            dfs.append(pd.read_parquet(f))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def train_model(train_df, test_df, target_col, features, model_name, drop_vals):
    # Filter out invalid labels (-1 timeout, -2 not triggered)
    df_tr = train_df[~train_df[target_col].isin(drop_vals)].copy()
    df_te = test_df[~test_df[target_col].isin(drop_vals)].copy()
    
    X_train = df_tr[features]
    y_train = df_tr[target_col]
    X_test = df_te[features]
    y_test = df_te[target_col]
    
    print(f"  Training shape: {X_train.shape[0]} | Testing shape: {X_test.shape[0]}")
    
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )
    
    return model

def evaluate_expectancy(model, eval_df, target_col, features, reward, risk, drop_vals, model_name):
    df_ev = eval_df[~eval_df[target_col].isin(drop_vals)].copy()
    if len(df_ev) == 0:
        return
        
    X_ev = df_ev[features]
    y_ev = df_ev[target_col].values
    
    probs = model.predict_proba(X_ev)[:, 1]
    
    thresholds = np.linspace(0.3, 0.7, 41)
    results = []
    
    base_wr = y_ev.mean()
    base_exp = (base_wr * reward) - ((1 - base_wr) * risk)
    
    for t in thresholds:
        signals = probs >= t
        taken_trades = signals.sum()
        if taken_trades == 0:
            continue
            
        wins = y_ev[signals].sum()
        wr = wins / taken_trades
        exp = (wr * reward) - ((1 - wr) * risk)
        results.append({'threshold': t, 'trades': taken_trades, 'win_rate': wr, 'expectancy': exp})
        
    res_df = pd.DataFrame(results)
    
    print(f"--- {model_name} Expectancy Evaluation on 2026 Holdout Set ---")
    print(f"Risk: {risk}R | Reward: {reward}R")
    print(f"Base Win Rate: {base_wr*100:.2f}% | Base Expectancy: {base_exp:.4f} R")
    
    if len(res_df) > 0:
        best_exp = res_df.loc[res_df['expectancy'].idxmax()]
        print(f"Best Threshold: {best_exp['threshold']:.2f} | Trades: {int(best_exp['trades'])} | WR: {best_exp['win_rate']*100:.2f}% | Max Exp: {best_exp['expectancy']:.4f} R")
        print("Top 5 Thresholds by Expectancy:")
        print(res_df.sort_values('expectancy', ascending=False).head(5).to_string(index=False))
    else:
        print("No thresholds triggered any trades.")
    print("\n" + "="*50 + "\n")

def main():
    data_dir = Path("outputs/sim_labels_v4")
    
    print("Loading datasets...")
    train_df = load_data([2020, 2021, 2022, 2023, 2024], data_dir)
    test_df = load_data([2025], data_dir)
    eval_df = load_data([2026], data_dir)
    
    features = [
        "time_sin", "time_cos", 
        "ema_50_5m_dist", "ema_200_5m_dist", "atr_14_5m", "return_12_5m",
        "ema_50_15m_dist", "ema_200_15m_dist", "atr_14_15m", "return_4_15m"
    ]
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # configs: (target_col, model_name, drop_vals, reward, risk)
    configs = [
        ("label_t1", "Model_T1", [-1], 1.0, 1.0),
        ("label_t2", "Model_T2", [-1, -2], 2.0, 1.0),
        ("label_t3", "Model_T3", [-1], 2.0, 1.0),
        ("label_t4", "Model_T4", [-1], 3.0, 1.0)
    ]
    
    print("\n" + "="*50)
    for target_col, name, drop_vals, reward, risk in configs:
        print(f"Training {name}...")
        model = train_model(train_df, test_df, target_col, features, name, drop_vals)
        model.save_model(models_dir / f"{name}.json")
        evaluate_expectancy(model, eval_df, target_col, features, reward, risk, drop_vals, name)

if __name__ == "__main__":
    main()
