"""
EXP-11: Adversarial Confidence Calibration
==========================================
Determines whether the Volume Fallback model's Prob_Win sigmoid output is
properly calibrated. Learns Temperature Scaling and Platt Scaling on the
validation set, evaluates them on the test set, and performs a joint threshold
sweep to find the optimal operating point for Expected Value.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.special import logit, expit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FALLBACK_DIR = BASE_DIR / "outputs" / "fallback"
TENSOR_DIR = FALLBACK_DIR / "tensors"
MODEL_PATH = FALLBACK_DIR / "model.keras"

EXP_DIR = BASE_DIR / "outputs" / "experiments" / "exp11_calibration"
EXP_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Prediction ──────────────────────────────────────────────────────
def safe_predict(model, micro, macro, batch_size=32):
    """Predict using manual batch loop to avoid model.predict() hangs on Mac Metal."""
    n_samples = len(micro)
    prob_wins = []
    pred_oss = []

    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        preds = model([micro[i:end], macro[i:end]], training=False)
        prob_wins.append(preds[0].numpy().flatten())
        pred_oss.append(preds[1].numpy().flatten())

    return np.concatenate(prob_wins), np.concatenate(pred_oss)


# ── 2. Calibration Metrics ─────────────────────────────────────────────
def compute_reliability(y_true, prob_pred, n_bins=20):
    """
    Computes reliability diagram data and Expected Calibration Error (ECE).
    Returns: bin_edges, bin_accuracies, bin_confidences, bin_counts, ECE
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (prob_pred >= bin_edges[i]) & (prob_pred <= bin_edges[i+1])
        else:
            mask = (prob_pred >= bin_edges[i]) & (prob_pred < bin_edges[i+1])
            
        count = mask.sum()
        bin_counts.append(count)
        
        if count > 0:
            bin_accuracies.append(y_true[mask].mean())
            bin_confidences.append(prob_pred[mask].mean())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_edges[i] + bin_edges[i+1]) / 2)

    total = len(y_true)
    ece = sum(
        (c / total) * abs(a - conf)
        for c, a, conf in zip(bin_counts, bin_accuracies, bin_confidences)
        if c > 0
    )
    return bin_edges, bin_accuracies, bin_confidences, bin_counts, ece


# ── 3. Calibration Learning ────────────────────────────────────────────
def learn_temperature(y_true, prob_pred):
    """Learns Temperature scalar T to minimize NLL."""
    raw = np.clip(prob_pred, 1e-7, 1 - 1e-7)
    logits = logit(raw)

    def nll(T):
        scaled = expit(logits / T)
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(scaled) + (1 - y_true) * np.log(1 - scaled))

    # Calculate NLL before calibration (T=1)
    nll_before = nll(1.0)

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    T_opt = result.x
    nll_after = result.fun

    return T_opt, nll_before, nll_after


def learn_platt(y_true, prob_pred):
    """Learns Platt Scaling parameters A and B via Logistic Regression."""
    raw = np.clip(prob_pred, 1e-7, 1 - 1e-7)
    logits = logit(raw).reshape(-1, 1)

    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(logits, y_true)

    A = float(lr.coef_[0, 0])
    B = float(lr.intercept_[0])
    return A, B


# ── 4. Threshold Sweeping & EV ─────────────────────────────────────────
def sweep_thresholds(y_true, prob_win, pred_os, label=""):
    """
    Sweeps thresholds on Prob_Win and Pred_OS.
    Returns the optimal parameters maximizing Expected Value (EV_total).
    Assumes 1:1 TP:SL ratio, meaning EV_per_trade = WR * 1.0 - (1-WR) * 1.0.
    """
    best_ev_total = -float('inf')
    best_params = {}
    
    # Sweep probabilities based on typical calibrated ranges
    prob_range = np.arange(0.4, 0.95, 0.02)
    os_range = np.arange(1.0, 3.0, 0.1)

    for th_prob in prob_range:
        for th_os in os_range:
            mask = (prob_win >= th_prob) & (pred_os >= th_os)
            n_trades = int(mask.sum())
            
            if n_trades > 0:
                wr = y_true[mask].mean()
                ev_per_trade = wr * 1.0 - (1 - wr) * 1.0
                ev_total = ev_per_trade * n_trades
            else:
                wr = 0.0
                ev_per_trade = 0.0
                ev_total = 0.0
                
            if ev_total > best_ev_total:
                best_ev_total = ev_total
                best_params = {
                    "threshold_prob": float(round(th_prob, 4)),
                    "threshold_os": float(round(th_os, 4)),
                    "wr": float(round(wr, 4)),
                    "n_trades": n_trades,
                    "ev_per_trade": float(round(ev_per_trade, 4)),
                    "ev_total": float(round(ev_total, 4))
                }
                
    # Also calculate the baseline point (0.6, 1.7) for comparison
    mask_base = (prob_win >= 0.6) & (pred_os >= 1.7)
    n_base = int(mask_base.sum())
    if n_base > 0:
        wr_base = y_true[mask_base].mean()
        ev_pt_base = wr_base * 1.0 - (1 - wr_base) * 1.0
        ev_tot_base = ev_pt_base * n_base
    else:
        wr_base, ev_pt_base, ev_tot_base = 0.0, 0.0, 0.0
        
    # Store the baseline specific metrics if this is the "raw" sweep
    if label == "raw":
        best_params = {
            "threshold_prob": 0.6,
            "threshold_os": 1.7,
            "wr": float(round(wr_base, 4)),
            "n_trades": n_base,
            "ev_per_trade": float(round(ev_pt_base, 4)),
            "ev_total": float(round(ev_tot_base, 4))
        }

    return best_params


# ── 5. Kelly Criterion Analysis ────────────────────────────────────────
def kelly_analysis(calibrated_prob, pred_os, y_true, threshold_prob, threshold_os):
    mask = (calibrated_prob >= threshold_prob) & (pred_os >= threshold_os)
    if mask.sum() == 0:
        return {
            "mean_kelly_fraction": 0.0,
            "mean_half_kelly": 0.0,
            "compounded_return_fixed": 0.0,
            "compounded_return_kelly": 0.0
        }
        
    trades_prob = calibrated_prob[mask]
    trades_true = y_true[mask]
    
    # f* = p - q (since b = 1 for 1:1 TP:SL)
    # where p = probability of win, q = 1 - p
    kelly_fractions = 2 * trades_prob - 1
    
    # Cap Kelly fractions at 0 to avoid shorting when the model is confident but p < 0.5 (shouldn't happen)
    kelly_fractions = np.maximum(kelly_fractions, 0)
    
    mean_kelly = float(kelly_fractions.mean())
    mean_half_kelly = mean_kelly / 2.0
    
    # Simulate compounding
    # Start with 1.0 capital
    cap_fixed = 1.0
    cap_kelly = 1.0
    
    # Fixed risk is typical: 1% per trade
    fixed_risk = 0.01
    
    for p_est, outcome in zip(kelly_fractions, trades_true):
        # Outcome is 1 (Win) or 0 (Loss)
        # Assuming 1:1 RR
        
        # Fixed 1% sizing
        if outcome == 1:
            cap_fixed *= (1 + fixed_risk)
        else:
            cap_fixed *= (1 - fixed_risk)
            
        # Half-Kelly sizing (capped at max 5% per trade for sanity)
        risk_fraction = min(p_est / 2.0, 0.05)
        if outcome == 1:
            cap_kelly *= (1 + risk_fraction)
        else:
            cap_kelly *= (1 - risk_fraction)
            
    return {
        "mean_kelly_fraction": float(round(mean_kelly, 4)),
        "mean_half_kelly": float(round(mean_half_kelly, 4)),
        "compounded_return_fixed": float(round(cap_fixed - 1.0, 4)),
        "compounded_return_kelly": float(round(cap_kelly - 1.0, 4))
    }


# ── Plotting Utilities ─────────────────────────────────────────────────
def plot_reliability_diagram(y_true, prob_pred, filename, title):
    edges, acc, conf, counts, ece = compute_reliability(y_true, prob_pred)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Filter out empty bins for plotting the line
    acc_clean = [a for a, c in zip(acc, counts) if c > 0]
    conf_clean = [c for c, count in zip(conf, counts) if count > 0]
    
    plt.plot(conf_clean, acc_clean, 's-', label=f'Model (ECE = {ece:.4f})', linewidth=2)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Empirical Win Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(EXP_DIR / filename, dpi=150)
    plt.close()


def plot_reliability_3way(y_true, prob_raw, prob_temp, prob_platt, filename):
    _, acc_raw, conf_raw, counts_raw, ece_raw = compute_reliability(y_true, prob_raw)
    _, acc_tmp, conf_tmp, counts_tmp, ece_tmp = compute_reliability(y_true, prob_temp)
    _, acc_plt, conf_plt, counts_plt, ece_plt = compute_reliability(y_true, prob_platt)

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    def filter_clean(acc, conf, counts):
        a_cln = [a for a, c in zip(acc, counts) if c > 0]
        c_cln = [c for c, count in zip(conf, counts) if count > 0]
        return c_cln, a_cln

    c_raw, a_raw = filter_clean(acc_raw, conf_raw, counts_raw)
    c_tmp, a_tmp = filter_clean(acc_tmp, conf_tmp, counts_tmp)
    c_plt, a_plt = filter_clean(acc_plt, conf_plt, counts_plt)

    plt.plot(c_raw, a_raw, 's-', label=f'Raw (ECE={ece_raw:.4f})', alpha=0.7)
    plt.plot(c_tmp, a_tmp, '^-', label=f'Temperature (ECE={ece_tmp:.4f})', alpha=0.8)
    plt.plot(c_plt, a_plt, 'o-', label=f'Platt (ECE={ece_plt:.4f})', alpha=0.8)

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Empirical Win Rate')
    plt.title('Test Set Reliability: Raw vs Calibrated')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(EXP_DIR / filename, dpi=150)
    plt.close()


def plot_prob_distributions(prob_raw, prob_temp, prob_platt, filename):
    plt.figure(figsize=(12, 6))
    plt.hist(prob_raw, bins=50, alpha=0.4, label='Raw', density=True)
    plt.hist(prob_temp, bins=50, alpha=0.4, label='Temperature', density=True)
    plt.hist(prob_platt, bins=50, alpha=0.4, label='Platt', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Test Set Probability Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(EXP_DIR / filename, dpi=150)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" EXP-11: Adversarial Confidence Calibration")
    print("=" * 60)

    # 1. Load Data & Model
    print("📂 Loading data and model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    val_micro = np.load(TENSOR_DIR / "val_micro.npy")
    val_macro = np.load(TENSOR_DIR / "val_macro.npy")
    val_y_class = np.load(TENSOR_DIR / "val_y_class.npy")

    test_micro = np.load(TENSOR_DIR / "test_micro.npy")
    test_macro = np.load(TENSOR_DIR / "test_macro.npy")
    test_y_class = np.load(TENSOR_DIR / "test_y_class.npy")
    test_y_mag = np.load(TENSOR_DIR / "test_y_mag.npy")

    print(f"   Validation samples: {len(val_micro):,}")
    print(f"   Test samples:       {len(test_micro):,}")

    # 2. Raw Predictions
    print("🔮 Generating predictions...")
    val_prob_raw, _ = safe_predict(model, val_micro, val_macro)
    test_prob_raw, test_pred_os = safe_predict(model, test_micro, test_macro)

    # 3. Validation ECE & Abort Check
    _, _, _, _, ece_val_raw = compute_reliability(val_y_class, val_prob_raw)
    print(f"📊 Validation Set ECE (Raw): {ece_val_raw:.4f}")
    
    abort_triggered = False
    if ece_val_raw < 0.02:
        print("⚠️  ECE < 0.02. Model is already well-calibrated. Proceeding for visualization only.")
        abort_triggered = True

    plot_reliability_diagram(val_y_class, val_prob_raw, "reliability_val_raw.png", 
                             f"Validation Set Reliability (Raw)")

    # 4. Learn Calibrations
    print("🧠 Learning calibration parameters...")
    T_opt, nll_before, nll_after = learn_temperature(val_y_class, val_prob_raw)
    A_opt, B_opt = learn_platt(val_y_class, val_prob_raw)
    
    print(f"   Temperature Scaling: T = {T_opt:.4f} (NLL: {nll_before:.4f} -> {nll_after:.4f})")
    print(f"   Platt Scaling:       A = {A_opt:.4f}, B = {B_opt:.4f}")

    # 5. Apply to Test Set
    test_raw_clipped = np.clip(test_prob_raw, 1e-7, 1 - 1e-7)
    test_logits = logit(test_raw_clipped)
    
    test_prob_temp = expit(test_logits / T_opt)
    test_prob_platt = expit(A_opt * test_logits + B_opt)

    # 6. Test Set ECEs
    _, _, _, _, ece_test_raw = compute_reliability(test_y_class, test_prob_raw)
    _, _, _, _, ece_test_temp = compute_reliability(test_y_class, test_prob_temp)
    _, _, _, _, ece_test_platt = compute_reliability(test_y_class, test_prob_platt)

    print("\n📊 Test Set ECE Comparison:")
    print(f"   Raw:         {ece_test_raw:.4f}")
    print(f"   Temperature: {ece_test_temp:.4f}")
    print(f"   Platt:       {ece_test_platt:.4f}")

    plot_reliability_3way(test_y_class, test_prob_raw, test_prob_temp, test_prob_platt, "reliability_test_3way.png")
    plot_prob_distributions(test_prob_raw, test_prob_temp, test_prob_platt, "prob_distribution.png")

    # Determine best calibration
    eces = {"none": ece_test_raw, "temperature": ece_test_temp, "platt": ece_test_platt}
    best_calib = min(eces, key=eces.get)

    # 7. Threshold Sweeps
    print("\n🔍 Sweeping thresholds for EV maximization...")
    sweep_raw = sweep_thresholds(test_y_class, test_prob_raw, test_pred_os, label="raw")
    sweep_temp = sweep_thresholds(test_y_class, test_prob_temp, test_pred_os)
    sweep_platt = sweep_thresholds(test_y_class, test_prob_platt, test_pred_os)

    print("   Baseline (Raw @ 0.6, 1.7):")
    print(f"     Trades: {sweep_raw['n_trades']}, WR: {sweep_raw['wr']:.2%}, EV_total: {sweep_raw['ev_total']:.2f}")
    
    print(f"   Optimal Temperature (@ {sweep_temp['threshold_prob']}, {sweep_temp['threshold_os']}):")
    print(f"     Trades: {sweep_temp['n_trades']}, WR: {sweep_temp['wr']:.2%}, EV_total: {sweep_temp['ev_total']:.2f}")
    
    print(f"   Optimal Platt (@ {sweep_platt['threshold_prob']}, {sweep_platt['threshold_os']}):")
    print(f"     Trades: {sweep_platt['n_trades']}, WR: {sweep_platt['wr']:.2%}, EV_total: {sweep_platt['ev_total']:.2f}")

    # 8. Kelly Analysis (on the best calibrated model)
    print("\n📈 Running Kelly Criterion analysis...")
    if best_calib == "temperature":
        best_prob = test_prob_temp
        best_th_p = sweep_temp["threshold_prob"]
        best_th_os = sweep_temp["threshold_os"]
    elif best_calib == "platt":
        best_prob = test_prob_platt
        best_th_p = sweep_platt["threshold_prob"]
        best_th_os = sweep_platt["threshold_os"]
    else:
        best_prob = test_prob_raw
        best_th_p = sweep_raw["threshold_prob"]
        best_th_os = sweep_raw["threshold_os"]
        
    kelly_res = kelly_analysis(best_prob, test_pred_os, test_y_class, best_th_p, best_th_os)
    print(f"   Mean Kelly fraction: {kelly_res['mean_kelly_fraction']:.4f}")
    print(f"   Fixed Sizing Return: {kelly_res['compounded_return_fixed']:.2%}")
    print(f"   Kelly Sizing Return: {kelly_res['compounded_return_kelly']:.2%}")

    # 9. Recommendations & Saving
    evs = {
        "keep_raw": sweep_raw["ev_total"],
        "adopt_temperature": sweep_temp["ev_total"] if ece_test_temp < ece_test_raw else -1,
        "adopt_platt": sweep_platt["ev_total"] if ece_test_platt < ece_test_raw else -1
    }
    
    rec = max(evs, key=evs.get)
    if abort_triggered:
        rec = "keep_raw"
        reason = "Validation ECE < 0.02. Model is already well calibrated."
    elif evs[rec] <= sweep_raw["ev_total"]:
        rec = "keep_raw"
        reason = "Calibration improved probabilities but did not yield higher Total EV."
    else:
        reason = f"Improves EV from {sweep_raw['ev_total']} to {evs[rec]} and lowers ECE."

    print(f"\n💡 Recommendation: {rec}")
    print(f"   Reason: {reason}")

    report = {
        "experiment_id": "EXP-11",
        "experiment_name": "Adversarial Confidence Calibration",
        "validation_set": {
            "n_samples": len(val_micro),
            "ECE_raw": round(ece_val_raw, 4),
            "abort_triggered": abort_triggered
        },
        "temperature_scaling": {
            "T_optimal": round(float(T_opt), 4),
            "NLL_before": round(float(nll_before), 4),
            "NLL_after": round(float(nll_after), 4)
        },
        "platt_scaling": {
            "A": round(A_opt, 4),
            "B": round(B_opt, 4)
        },
        "test_set": {
            "n_samples": len(test_micro),
            "ECE_raw": round(ece_test_raw, 4),
            "ECE_temperature": round(ece_test_temp, 4),
            "ECE_platt": round(ece_test_platt, 4),
            "best_calibration": best_calib
        },
        "threshold_comparison": {
            "raw": sweep_raw,
            "temperature_optimal": sweep_temp,
            "platt_optimal": sweep_platt
        },
        "kelly_analysis": kelly_res,
        "recommendation": rec,
        "recommendation_reason": reason
    }

    with open(EXP_DIR / "exp11_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(EXP_DIR / "calibration_params.json", "w") as f:
        json.dump({
            "T": float(T_opt),
            "A": float(A_opt),
            "B": float(B_opt)
        }, f, indent=2)
        
    if rec != "keep_raw":
        with open(EXP_DIR / "calibrated_config.json", "w") as f:
            if rec == "adopt_temperature":
                cfg = sweep_temp.copy()
            else:
                cfg = sweep_platt.copy()
            
            config_out = {
                "Prob_Win_threshold": cfg["threshold_prob"],
                "Pred_OS_threshold": cfg["threshold_os"],
                "z_score_window": 1000,
                "micro_buffer_size": 100,
                "macro_history_size": 10,
                "calibration": rec.split("_")[1]
            }
            json.dump(config_out, f, indent=2)

    # Markdown Analysis
    md_content = f"""# EXP-11 Analysis: Confidence Calibration

## Diagnostic
- **Validation ECE (Raw)**: {ece_val_raw:.4f}
- **Test ECE (Raw)**: {ece_test_raw:.4f}
{"- **Note**: Model was already highly calibrated (ECE < 0.02)." if abort_triggered else ""}

## Calibration Performance (Test Set)
- **Raw ECE**: {ece_test_raw:.4f}
- **Temperature ECE**: {ece_test_temp:.4f} (T = {T_opt:.4f})
- **Platt ECE**: {ece_test_platt:.4f} (A = {A_opt:.4f}, B = {B_opt:.4f})
- **Best Calibration**: {best_calib}

## Threshold Re-Sweep & Expected Value
Assuming a baseline 1:1 TP:SL ratio, maximizing Total EV (EV per trade × Number of trades):
- **Raw Baseline (0.6, 1.7)**: {sweep_raw['n_trades']} trades, {sweep_raw['wr']:.2%} WR -> **{sweep_raw['ev_total']:.2f} EV**
- **Optimal Temperature ({sweep_temp['threshold_prob']}, {sweep_temp['threshold_os']})**: {sweep_temp['n_trades']} trades, {sweep_temp['wr']:.2%} WR -> **{sweep_temp['ev_total']:.2f} EV**
- **Optimal Platt ({sweep_platt['threshold_prob']}, {sweep_platt['threshold_os']})**: {sweep_platt['n_trades']} trades, {sweep_platt['wr']:.2%} WR -> **{sweep_platt['ev_total']:.2f} EV**

## Risk & Kelly Sizing
Using the best calibrated model ({best_calib}):
- **Mean Kelly Fraction**: {kelly_res['mean_kelly_fraction']:.4f}
- Compounded Return (Fixed 1% risk): {kelly_res['compounded_return_fixed']:.2%}
- Compounded Return (Fractional Kelly, max 5%): {kelly_res['compounded_return_kelly']:.2%}

## Conclusion & Recommendation
**Decision**: {rec}
**Reason**: {reason}
"""
    with open(EXP_DIR / "exp11_analysis.md", "w") as f:
        f.write(md_content)

    print(f"\n💾 Saved all artifacts to {EXP_DIR}")


if __name__ == "__main__":
    main()
