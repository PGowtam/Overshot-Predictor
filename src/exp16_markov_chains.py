"""
EXP-16: Markov Chain Sequences
==============================
Builds empirical K-order transition matrices on the train split and applies 
Bayesian Fusion with the neural network outputs on the test split.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from scipy.special import logit, expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FALLBACK_DIR = BASE_DIR / "outputs" / "fallback"
TENSOR_DIR = FALLBACK_DIR / "tensors"
MODEL_PATH = FALLBACK_DIR / "model.keras"
LABELS_PATH = BASE_DIR / "outputs" / "exec" / "labels.parquet"

EXP11_DIR = BASE_DIR / "outputs" / "experiments" / "exp11_calibration"
EXP_DIR = BASE_DIR / "outputs" / "experiments" / "exp16_markov_chains"
EXP_DIR.mkdir(parents=True, exist_ok=True)

BASE_TH_PROB = 0.52
BASE_TH_OS = 1.0


def wilson_score(wins, total, z=1.96):
    if total == 0: return 0.5, 0.0, 1.0
    p = wins / total
    denominator = 1 + z**2 / total
    centre_adjusted_probability = p + z**2 / (2 * total)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return p, lower_bound, upper_bound


def assign_split(date):
    val_start = pd.Timestamp("2023-01-01", tz="UTC")
    test_start = pd.Timestamp("2023-07-01", tz="UTC")
    holdout_start = pd.Timestamp("2024-01-01", tz="UTC")
    if date < val_start: return "train"
    elif date < test_start: return "val"
    elif date < holdout_start: return "test"
    else: return "holdout"


def safe_predict(model, micro, macro, batch_size=32):
    n_samples = len(micro)
    prob_wins = []
    pred_oss = []
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        preds = model([micro[i:end], macro[i:end]], training=False)
        prob_wins.append(preds[0].numpy().flatten())
        pred_oss.append(preds[1].numpy().flatten())
    return np.concatenate(prob_wins), np.concatenate(pred_oss)


def compute_ev(mask, y_true):
    n_trades = int(mask.sum())
    if n_trades == 0: return 0.0, 0.0, 0
    wr = float(y_true[mask].mean())
    ev_per_trade = wr * 1.0 - (1.0 - wr) * 1.0
    return ev_per_trade * n_trades, wr, n_trades


def main():
    print("=" * 60)
    print(" EXP-16: Markov Chain Sequences")
    print("=" * 60)

    # 1. Load Global Labels
    print("📂 Loading global labels and resolving splits...")
    labels = pd.read_parquet(LABELS_PATH)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    n_bricks = len(labels)
    
    dirs = np.where(labels["uptrend"], 1, -1)
    y_class_global = labels["y_class"].values
    exclude_flags = labels["exclude_flag"].values
    
    splits = np.array([assign_split(d) for d in labels["date"]])
    
    # Replicate tensor_builder test index mapping
    test_indices = []
    for i in range(n_bricks):
        if i < 10: continue
        if bool(exclude_flags[i]): continue
        if pd.isna(y_class_global[i]): continue
        if splits[i] == "test":
            test_indices.append(i)
    test_indices = np.array(test_indices)
    
    # 2. Phase 1: Build Markov Tables (Train Split)
    print("⚙️  Building empirical K-order Markov models on train split...")
    K_VALUES = [3, 4, 5, 6, 7, 8]
    markov_tables = {k: {} for k in K_VALUES}
    
    train_mask = (splits == "train") & (~exclude_flags) & (~pd.isna(y_class_global))
    train_indices = np.where(train_mask)[0]
    
    for i in train_indices:
        if i < 8: continue # ensure we have enough history for max K
        
        current_dir = dirs[i]
        is_continuation = y_class_global[i] == 1.0
        
        for k in K_VALUES:
            # History of K bricks
            hist_dirs = dirs[i-k : i]
            # Relative encoding: 1 if same as current_dir, 0 if opposite
            pattern = "".join(["1" if d == current_dir else "0" for d in hist_dirs])
            
            if pattern not in markov_tables[k]:
                markov_tables[k][pattern] = {"total": 0, "wins": 0}
            
            markov_tables[k][pattern]["total"] += 1
            markov_tables[k][pattern]["wins"] += int(is_continuation)
            
    # Compute CIs
    structural_setups = {}
    structural_reversals = {}
    
    for k in K_VALUES:
        for pattern, stats in markov_tables[k].items():
            p, lb, ub = wilson_score(stats["wins"], stats["total"])
            stats["prob"] = p
            stats["ci_lower"] = lb
            stats["ci_upper"] = ub
            
            if stats["total"] >= 30: # min samples to be structural
                if lb > 0.6:
                    structural_setups[f"{k}_{pattern}"] = stats
                if ub < 0.4:
                    structural_reversals[f"{k}_{pattern}"] = stats

    print(f"  Found {len(structural_setups)} structural setups (CI LB > 0.6)")
    print(f"  Found {len(structural_reversals)} structural reversals (CI UB < 0.4)")
    
    with open(EXP_DIR / "markov_transition_tables.json", "w") as f:
        json.dump(markov_tables, f, indent=2)

    # 3. Phase 2: Bayesian Fusion on Test Split
    print("🏗️  Loading test tensors and model for Bayesian Fusion...")
    test_micro = np.load(TENSOR_DIR / "test_micro.npy")
    test_macro = np.load(TENSOR_DIR / "test_macro.npy")
    test_y_class = np.load(TENSOR_DIR / "test_y_class.npy")
    
    assert len(test_indices) == len(test_micro), "Alignment mismatch!"
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    raw_prob, pred_os = safe_predict(model, test_micro, test_macro)
    
    with open(EXP11_DIR / "calibration_params.json", "r") as f:
        T_opt = json.load(f)["T"]
        
    p_nn = expit(logit(np.clip(raw_prob, 1e-7, 1-1e-7)) / T_opt)
    
    base_mask = (p_nn >= BASE_TH_PROB) & (pred_os >= BASE_TH_OS)
    base_ev, base_wr, base_trades = compute_ev(base_mask, test_y_class)
    print(f"📊 Baseline (NN only): {base_trades} trades, {base_wr:.2%} WR, EV = {base_ev:.2f}")

    results = {}
    
    for k in K_VALUES:
        p_markov_arr = np.zeros(len(test_indices))
        
        for idx, global_i in enumerate(test_indices):
            current_dir = dirs[global_i]
            hist_dirs = dirs[global_i-k : global_i]
            pattern = "".join(["1" if d == current_dir else "0" for d in hist_dirs])
            
            if pattern in markov_tables[k] and markov_tables[k][pattern]["total"] > 10:
                # Use laplace smoothing
                w = markov_tables[k][pattern]["wins"]
                t = markov_tables[k][pattern]["total"]
                p_markov_arr[idx] = (w + 1) / (t + 2)
            else:
                p_markov_arr[idx] = 0.5 # Neutral prior if unseen or noisy
                
        # Bayesian Fusion
        p_fused = (p_nn * p_markov_arr) / (p_nn * p_markov_arr + (1 - p_nn) * (1 - p_markov_arr))
        
        # Test fused signal
        fused_mask = (p_fused >= BASE_TH_PROB) & (pred_os >= BASE_TH_OS)
        fused_ev, fused_wr, fused_trades = compute_ev(fused_mask, test_y_class)
        
        # Test structural veto (baseline + VETO if Markov CI UB < 0.4)
        veto_mask = base_mask.copy()
        for idx, global_i in enumerate(test_indices):
            if veto_mask[idx]:
                current_dir = dirs[global_i]
                hist_dirs = dirs[global_i-k : global_i]
                pattern = "".join(["1" if d == current_dir else "0" for d in hist_dirs])
                if f"{k}_{pattern}" in structural_reversals:
                    veto_mask[idx] = False
                    
        veto_ev, veto_wr, veto_trades = compute_ev(veto_mask, test_y_class)
        
        print(f"  K={k} Fused: EV={fused_ev:>6.2f} (WR={fused_wr:.2%}, Trades={fused_trades})")
        print(f"  K={k} Veto:  EV={veto_ev:>6.2f} (WR={veto_wr:.2%}, Trades={veto_trades})")
        
        results[k] = {
            "fused": {"ev": fused_ev, "wr": fused_wr, "trades": fused_trades},
            "veto": {"ev": veto_ev, "wr": veto_wr, "trades": veto_trades}
        }
        
    # Plotting EV Comparison
    plt.figure(figsize=(10,6))
    k_labels = [f"K={k}" for k in K_VALUES]
    fused_evs = [results[k]["fused"]["ev"] for k in K_VALUES]
    veto_evs = [results[k]["veto"]["ev"] for k in K_VALUES]
    
    x = np.arange(len(k_labels))
    width = 0.35
    
    plt.bar(x - width/2, fused_evs, width, label='Bayesian Fusion EV', color='blue')
    plt.bar(x + width/2, veto_evs, width, label='Structural Veto EV', color='orange')
    plt.axhline(base_ev, color='red', linestyle='--', label=f'NN Baseline EV ({base_ev:.2f})')
    
    plt.xticks(x, k_labels)
    plt.ylabel("Total Expected Value (R)")
    plt.title("EXP-16: Markov Chain Sequence EVs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(EXP_DIR / "exp16_fused_ev_comparison.png", dpi=150)
    plt.close()
    
    # Save Report
    with open(EXP_DIR / "exp16_report.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown Analysis
    md_content = f"""# EXP-16: Markov Chain Sequences (The Memory Model)

**Baseline (NN Only):** {base_trades} trades, {base_wr:.2%} WR, **{base_ev:.2f} EV**

## Structural Patterns (Train Set)
By computing K-order transition matrices on the train set (with N≥30 minimum occurrences), we identified:
- **{len(structural_setups)} Structural Setups** (95% CI Lower Bound > 0.6)
- **{len(structural_reversals)} Structural Reversals** (95% CI Upper Bound < 0.4)

## Test Set Fused Performance
We evaluated two methods on the unseen test set:
1. **Bayesian Fusion:** Mathematically blending the NN probability with the historical Markov probability.
2. **Structural Veto:** Keeping the NN's trade selections, but aggressively vetoing them if they trigger a structural reversal pattern.

| Model | Fusion EV | Fusion WR | Veto EV | Veto WR |
| :--- | :--- | :--- | :--- | :--- |
"""
    for k in K_VALUES:
        r = results[k]
        md_content += f"| K={k} | {r['fused']['ev']:.2f}R | {r['fused']['wr']:.2%} | {r['veto']['ev']:.2f}R | {r['veto']['wr']:.2%} |\n"
        
    md_content += """
## Conclusion
The results determine whether explicit sequence patterns contain predictive power beyond what the LSTM has already internalized.
"""
    with open(EXP_DIR / "exp16_analysis.md", "w") as f:
        f.write(md_content)
        
    print(f"\n💾 Saved all artifacts to {EXP_DIR}")


if __name__ == "__main__":
    main()
