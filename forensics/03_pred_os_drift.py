"""
Forensic Investigation #3: Pred_OS Calibration Drift Analysis

Tests whether Pred_OS thresholds (1.3, 1.6, 1.8) calibrated on 2023
validation data remain discriminative on 2024 holdout data.

Methodology:
  1. Load model and holdout tensors
  2. Generate prob_win and pred_os predictions
  3. Partition holdout by quarter (Q1-Q4 2024)
  4. For each quarter and each threshold:
     - Measure win rate of filtered trades
     - Measure trade count
     - Correlation between pred_os and actual y_mag
  5. Test: Does the prediction distribution shift over time?

Output:
  forensics/results/pred_os_drift.json
  forensics/results/pred_os_drift.png
"""

import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"
MODEL_PATH = OUTPUT_DIR / "model.keras"
CONFIG_PATH = OUTPUT_DIR / "config.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_BRICKS = 10


def safe_predict(model, micro, macro, batch_size=64):
    n = len(micro)
    pw, po = [], []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        preds = model([micro[i:end], macro[i:end]], training=False)
        pw.append(preds[0].numpy().flatten())
        po.append(preds[1].numpy().flatten())
    return np.concatenate(pw), np.concatenate(po)


def reconstruct_holdout_dates():
    """Reconstruct dates for holdout tensor samples."""
    holdout_path = OUTPUT_DIR / "holdout" / "labels.parquet"
    if not holdout_path.exists():
        return None
    df = pd.read_parquet(holdout_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    dates = []
    for i in range(len(df)):
        if i < CONTEXT_BRICKS:
            continue
        row = df.iloc[i]
        if "exclude_flag" in df.columns and bool(row["exclude_flag"]):
            continue
        if pd.isna(row["y_class"]):
            continue
        dates.append(row["date"])
    return np.array(dates)


def main():
    print("=" * 60)
    print(" FORENSIC #3: Pred_OS Calibration Drift Analysis")
    print("=" * 60)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load holdout tensors
    micro = np.load(TENSOR_DIR / "holdout_micro.npy")
    macro = np.load(TENSOR_DIR / "holdout_macro.npy")
    y_class = np.load(TENSOR_DIR / "holdout_y_class.npy")
    y_mag = np.load(TENSOR_DIR / "holdout_y_mag.npy")
    print(f"Loaded holdout: {len(micro):,} samples")

    # Get dates
    dates = reconstruct_holdout_dates()
    if dates is None or len(dates) != len(micro):
        print(f"⚠️  Date reconstruction mismatch: {len(dates) if dates is not None else 0} vs {len(micro)}")
        # Generate synthetic quarterly labels based on position
        n = len(micro)
        quarter_size = n // 4
        dates_quarter = np.concatenate([
            np.full(quarter_size, 1),
            np.full(quarter_size, 2),
            np.full(quarter_size, 3),
            np.full(n - 3*quarter_size, 4),
        ])
    else:
        dates_pd = pd.to_datetime(dates)
        dates_quarter = dates_pd.quarter

    # Predict
    print("Generating predictions...")
    prob_win, pred_os = safe_predict(model, micro, macro)

    # Load config for thresholds
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    th_prob = config["Prob_Win_threshold"]

    # Thresholds to test
    os_thresholds = [1.0, 1.1, 1.3, 1.6, 1.8, 2.0]

    # ── Overall baseline ────────────────────────────────────────
    print(f"\n📊 Overall Holdout Predictions:")
    print(f"   Pred_OS: mean={pred_os.mean():.4f}, std={pred_os.std():.4f}")
    print(f"   Pred_OS: P25={np.percentile(pred_os, 25):.4f}, P50={np.percentile(pred_os, 50):.4f}, "
          f"P75={np.percentile(pred_os, 75):.4f}")

    print(f"\n{'Threshold':>12} {'N Trades':>10} {'Win Rate':>10} {'Actual WR':>12}")
    print("-" * 50)
    for th_os in os_thresholds:
        mask = (prob_win >= th_prob) & (pred_os >= th_os)
        n_trades = np.sum(mask)
        wr = np.mean(y_class[mask]) if n_trades > 0 else 0
        print(f"  OS>={th_os:<6.1f} {n_trades:>10,} {wr:>10.4f}")

    # ── Quarterly breakdown ─────────────────────────────────────
    print("\n" + "=" * 60)
    print(" QUARTERLY DRIFT ANALYSIS")
    print("=" * 60)

    quarterly_results = {}
    quarters = sorted(np.unique(dates_quarter))

    for q in quarters:
        q_mask = dates_quarter == q
        q_pw = prob_win[q_mask]
        q_po = pred_os[q_mask]
        q_yc = y_class[q_mask]
        q_ym = y_mag[q_mask]
        n_q = np.sum(q_mask)

        print(f"\n  Q{q} 2024 (n={n_q:,}):")
        print(f"    Pred_OS: mean={q_po.mean():.4f}, std={q_po.std():.4f}")

        q_data = {
            "n_samples": int(n_q),
            "pred_os_mean": round(float(q_po.mean()), 4),
            "pred_os_std": round(float(q_po.std()), 4),
            "pred_os_median": round(float(np.median(q_po)), 4),
            "thresholds": {}
        }

        # Pearson r between pred_os and actual y_mag for wins
        win_mask = q_yc == 1
        if np.sum(win_mask) > 10:
            r = np.corrcoef(q_po[win_mask], q_ym[win_mask])[0, 1]
            q_data["pearson_r_wins"] = round(float(r), 4) if np.isfinite(r) else 0.0
            print(f"    Pearson r (Pred_OS vs y_mag, WINS): {r:.4f}")

        for th_os in os_thresholds:
            mask = (q_pw >= th_prob) & (q_po >= th_os)
            n_trades = int(np.sum(mask))
            wr = float(np.mean(q_yc[mask])) if n_trades > 0 else 0.0
            q_data["thresholds"][f"os_{th_os}"] = {
                "n_trades": n_trades,
                "win_rate": round(wr, 4)
            }
            print(f"    OS>={th_os:.1f}: {n_trades:>5} trades, WR={wr:.4f}")

        quarterly_results[f"Q{q}"] = q_data

    # ── KS Test: Q1 vs Q4 distribution shift ────────────────────
    from scipy.stats import ks_2samp
    q1_mask = dates_quarter == 1
    q4_mask = dates_quarter == 4

    if np.sum(q1_mask) > 10 and np.sum(q4_mask) > 10:
        ks_stat, ks_p = ks_2samp(pred_os[q1_mask], pred_os[q4_mask])
        print(f"\n📊 KS Test (Q1 vs Q4 Pred_OS distribution):")
        print(f"   KS statistic: {ks_stat:.4f}")
        print(f"   p-value: {ks_p:.6f}")
        print(f"   Significant shift: {'YES ⚠️' if ks_p < 0.05 else 'NO ✅'}")
        quarterly_results["ks_test_q1_q4"] = {
            "statistic": round(ks_stat, 4),
            "p_value": round(ks_p, 6),
            "significant": bool(ks_p < 0.05)
        }

    # ── Save ────────────────────────────────────────────────────
    with open(RESULTS_DIR / "pred_os_drift.json", "w") as f:
        json.dump(quarterly_results, f, indent=2)

    # ── Plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Pred_OS distribution by quarter
    for q in quarters:
        q_mask = dates_quarter == q
        axes[0, 0].hist(pred_os[q_mask], bins=50, alpha=0.4, label=f'Q{q}', density=True)
    axes[0, 0].set_xlabel('Pred_OS')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Pred_OS Distribution by Quarter', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: WR at different thresholds by quarter
    for th_os in [1.3, 1.6, 1.8]:
        wrs = []
        for q in quarters:
            key = f"Q{q}"
            if key in quarterly_results and f"os_{th_os}" in quarterly_results[key].get("thresholds", {}):
                wrs.append(quarterly_results[key]["thresholds"][f"os_{th_os}"]["win_rate"])
            else:
                wrs.append(0)
        axes[0, 1].plot(quarters, wrs, 'o-', label=f'OS>={th_os}', linewidth=2, markersize=6)
    axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Quarter (2024)')
    axes[0, 1].set_ylabel('Win Rate')
    axes[0, 1].set_title('Win Rate Stability by Quarter', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(quarters)

    # Plot 3: Trade count at different thresholds
    for th_os in [1.3, 1.6, 1.8]:
        counts = []
        for q in quarters:
            key = f"Q{q}"
            if key in quarterly_results and f"os_{th_os}" in quarterly_results[key].get("thresholds", {}):
                counts.append(quarterly_results[key]["thresholds"][f"os_{th_os}"]["n_trades"])
            else:
                counts.append(0)
        axes[1, 0].plot(quarters, counts, 's-', label=f'OS>={th_os}', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Quarter (2024)')
    axes[1, 0].set_ylabel('Number of Trades')
    axes[1, 0].set_title('Trade Frequency by Quarter', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(quarters)

    # Plot 4: Prob_Win vs Pred_OS scatter (sampled)
    n_plot = min(3000, len(prob_win))
    idx = np.random.choice(len(prob_win), n_plot, replace=False)
    colors = ['green' if y_class[i] == 1 else 'red' for i in idx]
    axes[1, 1].scatter(prob_win[idx], pred_os[idx], c=colors, alpha=0.2, s=5)
    axes[1, 1].axhline(y=1.3, color='blue', linestyle='--', alpha=0.5, label='OS=1.3')
    axes[1, 1].axhline(y=1.6, color='orange', linestyle='--', alpha=0.5, label='OS=1.6')
    axes[1, 1].axvline(x=th_prob, color='purple', linestyle='--', alpha=0.5, label=f'PW={th_prob}')
    axes[1, 1].set_xlabel('Prob_Win')
    axes[1, 1].set_ylabel('Pred_OS')
    axes[1, 1].set_title('Prediction Space (green=WIN, red=LOSS)', fontweight='bold')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pred_os_drift.png", dpi=150)
    print(f"\n💾 Saved: pred_os_drift.json, pred_os_drift.png")


if __name__ == "__main__":
    main()
