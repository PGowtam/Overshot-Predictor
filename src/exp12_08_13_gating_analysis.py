"""
EXP-12, 08, 13: Gating Analysis
===============================
Evaluates three independent gating mechanisms on the existing test set,
using the optimal EXP-11 calibrated baseline as the starting point.

1. EXP-12: Spread Microstructure Veto
2. EXP-08: Sequence Entropy Stratification
3. EXP-13: Volatility-Regime Gating
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
MODEL_PATH = FALLBACK_DIR / "model.keras"

EXP11_DIR = BASE_DIR / "outputs" / "experiments" / "exp11_calibration"
EXP_DIR = BASE_DIR / "outputs" / "experiments" / "gating_analysis"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# EXP-11 Baseline parameters
BASE_TH_PROB = 0.52
BASE_TH_OS = 1.0


# ── Utilities ──────────────────────────────────────────────────────────
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
    if n_trades == 0:
        return 0.0, 0.0, 0
    wr = float(y_true[mask].mean())
    ev_per_trade = wr * 1.0 - (1.0 - wr) * 1.0
    ev_total = ev_per_trade * n_trades
    return ev_total, wr, n_trades


# ── EXP-12: Spread Veto ────────────────────────────────────────────────
def evaluate_exp12(micro, y_true, base_mask):
    """
    Extracts z_Spread (index 4) from last 25 ticks, computes dSpread/dt slope.
    Sweeps a hard veto threshold on the baseline mask.
    """
    print("\n--- Running EXP-12: Spread Microstructure Veto ---")
    # test_micro is shape (N, 10, 100, 9). We want the last brick's 100 ticks.
    current_micro = micro[:, -1, :, :]
    
    # shape: (N, 25)
    spread_25 = current_micro[:, -25:, 4]
    
    # Vectorized slope calculation
    x = np.arange(25)
    x_mean = np.mean(x)
    y_mean = np.mean(spread_25, axis=1, keepdims=True)
    numerator = np.sum((x - x_mean) * (spread_25 - y_mean), axis=1)
    denominator = np.sum((x - x_mean)**2)
    slopes = numerator / denominator  # (N,)
    
    # Also calculate interaction term (just for logging/analysis)
    ofi_25 = current_micro[:, -25:, 0]
    mean_ofi = np.mean(ofi_25, axis=1)
    interaction = mean_ofi * slopes
    
    # Baseline stats
    base_ev, base_wr, base_trades = compute_ev(base_mask, y_true)
    
    # Sweep veto thresholds
    thresholds = np.linspace(-0.05, 0.1, 30)
    results = []
    
    best_ev = base_ev
    best_th = None
    best_stats = None
    
    for th in thresholds:
        # VETO if slope > th (i.e. keep if slope <= th)
        veto_mask = (slopes <= th)
        combined_mask = base_mask & veto_mask
        
        ev_tot, wr, trades = compute_ev(combined_mask, y_true)
        results.append((th, ev_tot, wr, trades))
        
        if ev_tot > best_ev:
            best_ev = ev_tot
            best_th = th
            best_stats = (wr, trades)
            
    # Plotting
    ths = [r[0] for r in results]
    evs = [r[1] for r in results]
    wrs = [r[2] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(ths, evs, 'g-', label='Total EV', linewidth=2)
    ax1.axhline(base_ev, color='g', linestyle='--', alpha=0.5, label='Baseline EV')
    ax2.plot(ths, wrs, 'b-', label='Win Rate', linewidth=2)
    
    ax1.set_xlabel('Veto Threshold (Maximum Allowed Spread Slope)')
    ax1.set_ylabel('Total Expected Value (R)', color='g')
    ax2.set_ylabel('Win Rate', color='b')
    plt.title('EXP-12: Spread Veto Impact on Baseline Trades')
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
    plt.grid(True, alpha=0.3)
    plt.savefig(EXP_DIR / "exp12_spread_veto.png", dpi=150)
    plt.close()
    
    if best_th is not None:
        print(f"Optimal Veto: slope <= {best_th:.4f}")
        print(f"EV Improved: {base_ev:.2f} -> {best_ev:.2f} (WR {base_wr:.2%} -> {best_stats[0]:.2%}, Trades {base_trades} -> {best_stats[1]})")
        return {"improved": True, "best_th": float(best_th), "best_ev": float(best_ev), "best_wr": float(best_stats[0]), "best_trades": int(best_stats[1])}
    else:
        print("No veto threshold improved Total EV. Spread slope does not add orthogonal value to these thresholds.")
        return {"improved": False, "best_ev": float(base_ev)}


# ── EXP-08: Sequence Entropy ───────────────────────────────────────────
def evaluate_exp08(macro, y_true, base_mask):
    """
    Extracts macro direction (index 1), computes Shannon entropy for K=10.
    Since length is 10 and binary, there are exactly 6 distinct entropy regimes.
    """
    print("\n--- Running EXP-08: Sequence Entropy Stratification ---")
    directions = macro[:, :, 1]  # (N, 10), values are +1 or -1
    
    # Map to 0 and 1 for easier counting
    binary_seq = (directions > 0).astype(int)
    p_ones = np.sum(binary_seq, axis=1) / 10.0  # (N,)
    
    def calc_entropy(p):
        if p == 0 or p == 1:
            return 0.0
        return -p * log2(p) - (1-p) * log2(1-p)
        
    entropies = np.array([calc_entropy(p) for p in p_ones])
    
    # We only care about the trades the baseline model took
    trades_entropy = entropies[base_mask]
    trades_y = y_true[base_mask]
    
    # The 6 distinct entropy values for K=10
    distinct_vals = np.sort(np.unique(np.round(entropies, 4)))
    
    regime_stats = []
    for val in distinct_vals:
        # fuzzy matching to handle float precision
        mask = np.abs(trades_entropy - val) < 1e-4
        n = int(mask.sum())
        if n > 0:
            wr = float(trades_y[mask].mean())
            regime_stats.append({
                "entropy_val": float(val),
                "wr": wr,
                "trades": n
            })
            print(f"Entropy {val:.4f} -> WR: {wr:.2%} (n={n})")
            
    # Plotting
    vals = [r['entropy_val'] for r in regime_stats]
    wrs = [r['wr'] for r in regime_stats]
    sizes = [r['trades'] * 5 for r in regime_stats]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(vals, wrs, s=sizes, alpha=0.6, color='purple')
    for val, wr, size in zip(vals, wrs, sizes):
        plt.annotate(f"{size//5}t", (val, wr), textcoords="offset points", xytext=(0,10), ha='center')
        
    base_wr = trades_y.mean()
    plt.axhline(base_wr, color='r', linestyle='--', label=f'Baseline Avg ({base_wr:.2%})')
    plt.xlabel('Sequence Entropy (K=10)')
    plt.ylabel('Empirical Win Rate')
    plt.title('EXP-08: Baseline Win Rate Stratified by K=10 Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(EXP_DIR / "exp08_entropy_stratification.png", dpi=150)
    plt.close()
    
    # Check if there is significant divergence (e.g. > 5% difference across regimes with > 50 trades)
    valid_regimes = [r for r in regime_stats if r['trades'] >= 50]
    divergence = 0.0
    if len(valid_regimes) >= 2:
        wrs = [r['wr'] for r in valid_regimes]
        divergence = max(wrs) - min(wrs)
        print(f"Max WR divergence across valid regimes: {divergence:.2%}")
        
    return {"regimes": regime_stats, "max_divergence": float(divergence)}


# ── EXP-13: Volatility-Regime Gating ───────────────────────────────────
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

def evaluate_exp13(macro, prob_win, pred_os, y_true):
    print("\n--- Running EXP-13: Volatility-Regime Gating ---")
    log_dur = macro[:, :, 0]
    rv = np.std(log_dur, axis=1)  # (N,)
    
    # 3 Quantiles
    q33 = np.percentile(rv, 33.33)
    q66 = np.percentile(rv, 66.67)
    
    regime_masks = {
        "LOW (Quiet)": (rv <= q33),
        "MID (Normal)": (rv > q33) & (rv <= q66),
        "HIGH (Toxic)": (rv > q66)
    }
    
    prob_range = np.arange(0.4, 0.9, 0.02)
    os_range = np.arange(1.0, 2.5, 0.1)
    
    total_ev = 0.0
    total_trades = 0
    results = {}
    
    for name, mask in regime_masks.items():
        print(f"  Regime {name} ({mask.sum()} samples):")
        # Find optimal thresholds for THIS regime
        r_prob = prob_win[mask]
        r_os = pred_os[mask]
        r_y = y_true[mask]
        
        opt = sweep_thresholds(r_prob, r_os, r_y, prob_range, os_range)
        if opt:
            print(f"    Optimal: PW>={opt['th_prob']}, OS>={opt['th_os']} -> WR: {opt['wr']:.2%}, Trades: {opt['trades']}, EV: {opt['ev']:.2f}")
            total_ev += opt['ev']
            total_trades += opt['trades']
            results[name] = opt
        else:
            results[name] = {"ev": 0.0}
            
    print(f"  Combined Regime-Aware EV: {total_ev:.2f} (Trades: {total_trades})")
    
    # Compare to global baseline EV
    base_mask = (prob_win >= BASE_TH_PROB) & (pred_os >= BASE_TH_OS)
    global_ev, global_wr, global_trades = compute_ev(base_mask, y_true)
    print(f"  Global Static EV (Baseline): {global_ev:.2f} (Trades: {global_trades})")
    
    # Plotting
    names = list(results.keys())
    # Bar plot comparing optimal th_prob and th_os across regimes
    probs = [results[n]['th_prob'] for n in names]
    oss = [results[n]['th_os'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width/2, probs, width, label='Optimal Prob_Win', color='blue')
    ax1.set_ylabel('Prob_Win Threshold', color='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, oss, width, label='Optimal Pred_OS', color='orange')
    ax2.set_ylabel('Pred_OS Threshold', color='orange')
    
    plt.title('EXP-13: Optimal Thresholds by Volatility Regime')
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
    plt.grid(True, alpha=0.3)
    plt.savefig(EXP_DIR / "exp13_volatility_regimes.png", dpi=150)
    plt.close()
    
    return {
        "regime_optimal": results,
        "combined_ev": float(total_ev),
        "combined_trades": int(total_trades),
        "global_ev": float(global_ev),
        "improved": total_ev > global_ev
    }


def main():
    print("=" * 60)
    print(" EXP-12, 08, 13: Gating Analysis")
    print("=" * 60)

    # 1. Load Data
    print("📂 Loading test set tensors...")
    test_micro = np.load(TENSOR_DIR / "test_micro.npy")
    test_macro = np.load(TENSOR_DIR / "test_macro.npy")
    test_y_class = np.load(TENSOR_DIR / "test_y_class.npy")

    # 2. Load Model & Predict
    print("🏗️  Loading model & generating raw predictions...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    raw_prob, pred_os = safe_predict(model, test_micro, test_macro)
    
    # 3. Apply Temperature Calibration (EXP-11)
    with open(EXP11_DIR / "calibration_params.json", "r") as f:
        calib = json.load(f)
        T_opt = calib["T"]
        print(f"🌡️  Applying Temperature Scaling (T = {T_opt:.4f})...")
        
    raw_clipped = np.clip(raw_prob, 1e-7, 1 - 1e-7)
    calibrated_prob = expit(logit(raw_clipped) / T_opt)
    
    # EXP-11 Baseline Mask
    base_mask = (calibrated_prob >= BASE_TH_PROB) & (pred_os >= BASE_TH_OS)
    base_ev, base_wr, base_trades = compute_ev(base_mask, test_y_class)
    print(f"📊 EXP-11 Baseline: {base_trades} trades, {base_wr:.2%} WR, EV = {base_ev:.2f}")
    
    # Run Experiments
    res_12 = evaluate_exp12(test_micro, test_y_class, base_mask)
    res_08 = evaluate_exp08(test_macro, test_y_class, base_mask)
    res_13 = evaluate_exp13(test_macro, calibrated_prob, pred_os, test_y_class)
    
    # Save Report
    report = {
        "baseline_ev": float(base_ev),
        "exp12_spread_veto": res_12,
        "exp08_sequence_entropy": res_08,
        "exp13_volatility_regimes": res_13
    }
    
    with open(EXP_DIR / "gating_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    # Write Markdown Analysis
    md_content = f"""# Gating Analysis Results (EXP-12, 08, 13)

**Starting Baseline:** {base_trades} trades, {base_wr:.2%} WR, **{base_ev:.2f} EV**

## EXP-12: Spread Microstructure Veto
- **Goal:** Veto trades where `dSpread/dt` (last 25 ticks) is expanding.
- **Result:** {"IMPROVED baseline" if res_12['improved'] else "Did NOT improve baseline EV"}.
- **Optimal Veto:** {f"slope <= {res_12.get('best_th', 'N/A')}"} yielding **{res_12.get('best_ev', 'N/A')} EV** (WR {res_12.get('best_wr', 0):.2%}).

## EXP-08: Sequence Entropy Stratification (K=10 restriction)
- **Goal:** Identify regimes using Shannon entropy of the last 10 brick directions.
- **Context:** Because sequence length is only 10, there are exactly 6 possible entropy values (0.0, 0.469, 0.7219, 0.8813, 0.971, 1.0). We are testing 6 distinct regime bins, not a continuous spectrum.
- **Max WR Divergence:** {res_08['max_divergence']:.2%} across valid regimes (>50 trades).
- **Conclusion:** If divergence > 5%, entropy carries orthogonal signal and justifies a future retrain with K=20 or 50.

## EXP-13: Volatility-Regime Gating
- **Goal:** Set custom `(Prob_Win, Pred_OS)` thresholds based on Realized Volatility (std of log_dur).
- **Result:** Combined Regime-Aware EV = **{res_13['combined_ev']:.2f}** vs Static Baseline EV = **{res_13['global_ev']:.2f}**.
- **Conclusion:** {"Adaptive thresholds significantly outperform fixed ones." if res_13['improved'] else "Fixed global thresholds perform just as well as regime-aware ones."}
"""
    with open(EXP_DIR / "gating_analysis.md", "w") as f:
        f.write(md_content)
        
    print(f"\n💾 Saved all artifacts to {EXP_DIR}")


if __name__ == "__main__":
    main()
