"""
EXP-12, 08, 13: Deep Dive Gating Analysis
=========================================
Uses global test indices to query K=50 historical macro vectors instantly,
evaluating gating logic at unprecedented depth.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from scipy.special import logit, expit
from math import log2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FALLBACK_DIR = BASE_DIR / "outputs" / "fallback"
TENSOR_DIR = FALLBACK_DIR / "tensors"
FEATURE_DIR = FALLBACK_DIR / "features"
MODEL_PATH = FALLBACK_DIR / "model.keras"
LABELS_PATH = BASE_DIR / "outputs" / "exec" / "labels.parquet"

EXP11_DIR = BASE_DIR / "outputs" / "experiments" / "exp11_calibration"
EXP_DIR = BASE_DIR / "outputs" / "experiments" / "gating_deep_dive"
EXP_DIR.mkdir(parents=True, exist_ok=True)

BASE_TH_PROB = 0.52
BASE_TH_OS = 1.0


# ── Data Alignment ─────────────────────────────────────────────────────
def get_test_global_indices():
    """Replicates tensor_builder's split logic to find global indices of test set."""
    labels = pd.read_parquet(LABELS_PATH)
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    n_bricks = len(labels)
    
    val_start = pd.Timestamp("2023-01-01", tz="UTC")
    test_start = pd.Timestamp("2023-07-01", tz="UTC")
    holdout_start = pd.Timestamp("2024-01-01", tz="UTC")
    
    def assign_split(d):
        if d < val_start: return "train"
        elif d < test_start: return "val"
        elif d < holdout_start: return "test"
        else: return "holdout"

    test_indices = []
    skipped_context = 0
    skipped_exclude = 0
    skipped_no_label = 0
    
    for i in range(n_bricks):
        if i < 10:
            skipped_context += 1
            continue
        row = labels.iloc[i]
        if bool(row["exclude_flag"]):
            skipped_exclude += 1
            continue
        if pd.isna(row["y_class"]):
            skipped_no_label += 1
            continue
            
        split = assign_split(row["date"])
        
        # training fast brick exclusion is only for train split
        if split == "test":
            test_indices.append(i)
            
    return np.array(test_indices)


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

def sweep_thresholds(prob, os, y_true, prob_range, os_range):
    best_ev = -float('inf')
    best_params = None
    for th_prob in prob_range:
        for th_os in os_range:
            mask = (prob >= th_prob) & (os >= th_os)
            ev, wr, trades = compute_ev(mask, y_true)
            if ev > best_ev:
                best_ev = ev
                best_params = {"th_prob": round(th_prob, 2), "th_os": round(th_os, 1), "ev": ev, "wr": wr, "trades": trades}
    return best_params


# ── EXP-12: Deep Dive Spread Microstructure ────────────────────────────
def evaluate_exp12_deep(micro, y_true, base_mask):
    print("\n--- Running EXP-12: Spread Deep Dive ---")
    current_micro = micro[:, -1, :, :]  # (N, 100, 9)
    spread_25 = current_micro[:, -25:, 4]
    
    x = np.arange(25)
    x_mean = np.mean(x)
    y_mean = np.mean(spread_25, axis=1, keepdims=True)
    numerator = np.sum((x - x_mean) * (spread_25 - y_mean), axis=1)
    denominator = np.sum((x - x_mean)**2)
    slopes = numerator / denominator  # (N,)
    
    ofi_25 = current_micro[:, -25:, 0]
    mean_ofi = np.mean(ofi_25, axis=1)
    interaction = mean_ofi * slopes
    
    base_ev, base_wr, base_trades = compute_ev(base_mask, y_true)
    
    # 1. Dynamic 3-state categorization
    print("  Sweeping Dynamic Spread Regimes...")
    thresholds = np.linspace(0.005, 0.05, 20)
    best_divergence = 0
    best_th = 0
    best_stats = {}
    
    # We only analyze trades taken by the baseline
    trades_slopes = slopes[base_mask]
    trades_y = y_true[base_mask]
    
    for th in thresholds:
        m_contracting = trades_slopes < -th
        m_stable = (trades_slopes >= -th) & (trades_slopes <= th)
        m_expanding = trades_slopes > th
        
        wr_con = trades_y[m_contracting].mean() if m_contracting.sum() > 0 else 0
        wr_exp = trades_y[m_expanding].mean() if m_expanding.sum() > 0 else 0
        
        div = wr_con - wr_exp
        if div > best_divergence and m_contracting.sum() > 20 and m_expanding.sum() > 20:
            best_divergence = div
            best_th = th
            best_stats = {
                "contracting": {"wr": float(wr_con), "n": int(m_contracting.sum())},
                "stable": {"wr": float(trades_y[m_stable].mean() if m_stable.sum()>0 else 0), "n": int(m_stable.sum())},
                "expanding": {"wr": float(wr_exp), "n": int(m_expanding.sum())}
            }
            
    print(f"    Optimal Regime Threshold: |slope| = {best_th:.4f}")
    if best_divergence > 0:
        print(f"    Contracting WR: {best_stats['contracting']['wr']:.2%} (n={best_stats['contracting']['n']})")
        print(f"    Stable WR:      {best_stats['stable']['wr']:.2%} (n={best_stats['stable']['n']})")
        print(f"    Expanding WR:   {best_stats['expanding']['wr']:.2%} (n={best_stats['expanding']['n']})")
        print(f"    Max Divergence: {best_divergence:.2%}")
    else:
        print("    No significant divergence found across dynamic spread regimes.")

    # 2. Interaction Term Analysis (z_OFI * dSpread/dt)
    print("  Analyzing Interaction Term (z_OFI * dSpread/dt)...")
    trades_interaction = interaction[base_mask]
    
    # Stratify by quartiles
    q_vals = np.percentile(trades_interaction, [25, 50, 75])
    m_q1 = trades_interaction <= q_vals[0]
    m_q2 = (trades_interaction > q_vals[0]) & (trades_interaction <= q_vals[1])
    m_q3 = (trades_interaction > q_vals[1]) & (trades_interaction <= q_vals[2])
    m_q4 = trades_interaction > q_vals[2]
    
    wr_q1 = trades_y[m_q1].mean()
    wr_q4 = trades_y[m_q4].mean()
    print(f"    Q1 (Most Negative, Conviction): {wr_q1:.2%} (n={m_q1.sum()})")
    print(f"    Q4 (Most Positive, Divergence): {wr_q4:.2%} (n={m_q4.sum()})")
    
    return {
        "dynamic_best_th": float(best_th),
        "dynamic_divergence": float(best_divergence),
        "dynamic_stats": best_stats,
        "interaction_q1_wr": float(wr_q1),
        "interaction_q4_wr": float(wr_q4)
    }

# ── EXP-08: Deep Dive Sequence Entropy & RLE ───────────────────────────
def compute_run_length(seq):
    if len(seq) == 0: return 0.0
    runs = []
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    return np.mean(runs)

def evaluate_exp08_deep(global_indices, macro_vectors, y_true, base_mask):
    print("\n--- Running EXP-08: Sequence Entropy & RLE Deep Dive ---")
    trades_global_idx = global_indices[base_mask]
    trades_y = y_true[base_mask]
    
    results = {}
    
    for K in [10, 15, 20, 30, 50]:
        print(f"  Evaluating K={K}...")
        entropies = []
        rles = []
        for idx in trades_global_idx:
            # slice last K macro vectors
            start_idx = max(0, idx - K + 1)
            macro_k = macro_vectors[start_idx : idx + 1]
            dirs = macro_k[:, 1]
            
            # Entropy
            p = np.sum(dirs > 0) / len(dirs)
            if p == 0 or p == 1: e = 0.0
            else: e = -p * log2(p) - (1-p) * log2(1-p)
            entropies.append(e)
            
            # RLE
            rles.append(compute_run_length(dirs > 0))
            
        entropies = np.array(entropies)
        rles = np.array(rles)
        
        # Quartile stratification for Entropy
        q = np.percentile(entropies, [25, 50, 75])
        q1_wr = trades_y[entropies <= q[0]].mean()
        q4_wr = trades_y[entropies > q[2]].mean()
        div = q1_wr - q4_wr
        print(f"    Entropy Q1 (Trend) WR: {q1_wr:.2%} | Q4 (Chop) WR: {q4_wr:.2%} | Div: {div:.2%}")
        
        # Save plot for each K
        plt.figure(figsize=(8,5))
        plt.scatter(entropies, trades_y, alpha=0.1, color='purple')
        # We can't scatter beautifully with binary Y. Let's do decile bar plots.
        deciles = np.percentile(entropies, np.arange(10, 101, 10))
        decile_wrs = []
        prev = -1.0
        valid_deciles = []
        for d in deciles:
            m = (entropies > prev) & (entropies <= d)
            if m.sum() > 0:
                decile_wrs.append(trades_y[m].mean())
                valid_deciles.append(f"{d:.2f}")
            prev = d
            
        plt.clf()
        plt.bar(valid_deciles, decile_wrs, color='purple', alpha=0.7)
        plt.axhline(trades_y.mean(), color='r', linestyle='--')
        plt.title(f"EXP-08: Win Rate by Entropy Decile (K={K})")
        plt.xlabel("Entropy Upper Bound")
        plt.ylabel("Empirical Win Rate")
        plt.ylim(0.5, 1.0)
        plt.savefig(EXP_DIR / f"exp08_entropy_K{K}.png", dpi=100)
        plt.close()
        
        res = {"entropy_div": float(div)}
        
        if K in [20, 50]:
            # Quartile stratification for RLE
            rq = np.percentile(rles, [25, 50, 75])
            rq1_wr = trades_y[rles <= rq[0]].mean()
            rq4_wr = trades_y[rles > rq[2]].mean()
            r_div = rq4_wr - rq1_wr # High RLE (Trend) should have higher WR
            print(f"    RLE Q4 (High Trend) WR: {rq4_wr:.2%} | Q1 (High Chop) WR: {rq1_wr:.2%} | Div: {r_div:.2%}")
            res["rle_div"] = float(r_div)
            
        results[f"K={K}"] = res
        
    return results

# ── EXP-13: Deep Dive Volatility Regimes ───────────────────────────────
def evaluate_exp13_deep(global_indices, macro_vectors, prob_win, pred_os, y_true):
    print("\n--- Running EXP-13: Volatility Deep Dive ---")
    prob_range = np.arange(0.4, 0.9, 0.02)
    os_range = np.arange(1.0, 2.5, 0.1)
    
    results = {}
    
    for K in [20, 50]:
        print(f"  Evaluating K={K}...")
        rvs = []
        for idx in global_indices:
            start_idx = max(0, idx - K + 1)
            macro_k = macro_vectors[start_idx : idx + 1]
            log_dur = macro_k[:, 0]
            rvs.append(np.std(log_dur))
            
        rvs = np.array(rvs)
        
        q33 = np.percentile(rvs, 33.33)
        q66 = np.percentile(rvs, 66.67)
        
        regime_masks = {
            "LOW (Quiet)": (rvs <= q33),
            "MID (Normal)": (rvs > q33) & (rvs <= q66),
            "HIGH (Toxic)": (rvs > q66)
        }
        
        total_ev = 0.0
        total_trades = 0
        k_results = {}
        
        for name, mask in regime_masks.items():
            r_prob = prob_win[mask]
            r_os = pred_os[mask]
            r_y = y_true[mask]
            
            opt = sweep_thresholds(r_prob, r_os, r_y, prob_range, os_range)
            if opt:
                print(f"    {name}: PW>={opt['th_prob']}, OS>={opt['th_os']} -> WR: {opt['wr']:.2%}, EV: {opt['ev']:.2f}")
                total_ev += opt['ev']
                total_trades += opt['trades']
                k_results[name] = opt
            else:
                k_results[name] = {"ev": 0.0}
                
        print(f"    Combined EV for K={K}: {total_ev:.2f}")
        k_results["combined_ev"] = float(total_ev)
        results[f"K={K}"] = k_results
        
    return results

def main():
    print("=" * 60)
    print(" Deep Dive Gating Analysis (EXP-12, 08, 13)")
    print("=" * 60)

    print("📂 Aligning Test Tensors to Global Macro DB...")
    global_indices = get_test_global_indices()
    macro_vectors = np.load(FEATURE_DIR / "macro_vectors.npy")
    test_micro = np.load(TENSOR_DIR / "test_micro.npy")
    test_y_class = np.load(TENSOR_DIR / "test_y_class.npy")
    
    assert len(global_indices) == len(test_micro), f"Alignment mismatch: {len(global_indices)} != {len(test_micro)}"
    
    print("🏗️  Loading model & generating raw predictions...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # We load test_macro just for prediction
    test_macro = np.load(TENSOR_DIR / "test_macro.npy")
    raw_prob, pred_os = safe_predict(model, test_micro, test_macro)
    
    with open(EXP11_DIR / "calibration_params.json", "r") as f:
        calib = json.load(f)
        T_opt = calib["T"]
        
    raw_clipped = np.clip(raw_prob, 1e-7, 1 - 1e-7)
    calibrated_prob = expit(logit(raw_clipped) / T_opt)
    
    base_mask = (calibrated_prob >= BASE_TH_PROB) & (pred_os >= BASE_TH_OS)
    base_ev, base_wr, base_trades = compute_ev(base_mask, test_y_class)
    print(f"📊 Baseline: {base_trades} trades, {base_wr:.2%} WR, EV = {base_ev:.2f}")
    
    # Exec
    res_12 = evaluate_exp12_deep(test_micro, test_y_class, base_mask)
    res_08 = evaluate_exp08_deep(global_indices, macro_vectors, test_y_class, base_mask)
    res_13 = evaluate_exp13_deep(global_indices, macro_vectors, calibrated_prob, pred_os, test_y_class)
    
    report = {
        "baseline_ev": float(base_ev),
        "exp12_deep": res_12,
        "exp08_deep": res_08,
        "exp13_deep": res_13
    }
    
    with open(EXP_DIR / "deep_gating_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n💾 Saved all artifacts to {EXP_DIR}")


if __name__ == "__main__":
    main()
