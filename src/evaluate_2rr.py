"""
2RR Overshot: Evaluation
=========================
Evaluates the 2RR model on the test set and compares:
  1. 2RR model (trained to predict 2-brick continuation)
  2. Fallback model (trained for 1RR, used as baseline)

The key metric is: can the 2RR model achieve WR > 33.3% on the test set
with enough trades to be meaningful?

Reads from:
  - outputs/fallback_2rr/{model.keras, config.json, tensors/}
  - outputs/fallback/{model.keras, config.json, tensors/}  (1RR baseline)

Outputs:
  - Console report
  - outputs/fallback_2rr/evaluation_report.json
  - outputs/fallback_2rr/plots/evaluation_bar.png
"""

import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

OUTPUT_DIR = BASE_DIR / "outputs"

# 2RR paths
RR2_DIR = OUTPUT_DIR / "fallback_2rr"
RR2_TENSOR_DIR = RR2_DIR / "tensors"
RR2_MODEL_PATH = RR2_DIR / "model.keras"
RR2_CONFIG_PATH = RR2_DIR / "config.json"
RR2_PLOT_DIR = RR2_DIR / "plots"

# 1RR fallback paths (baseline comparison)
FB_DIR = OUTPUT_DIR / "fallback"
FB_TENSOR_DIR = FB_DIR / "tensors"
FB_MODEL_PATH = FB_DIR / "model.keras"
FB_CONFIG_PATH = FB_DIR / "config.json"

RR2_PLOT_DIR.mkdir(parents=True, exist_ok=True)

BREAKEVEN_WR = 1.0 / 3.0


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


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


def evaluate():
    # ── Load 2RR Model ──────────────────────────────────────────
    if not RR2_MODEL_PATH.exists():
        print("❌ 2RR model not found.")
        return
    if not RR2_CONFIG_PATH.exists():
        print("❌ 2RR config not found.")
        return

    rr2_config = load_config(RR2_CONFIG_PATH)
    th_prob = rr2_config["Prob_2RR_threshold"]
    th_os = rr2_config["Pred_OS_threshold"]
    print(f"⚙️  2RR Thresholds: Prob_2RR >= {th_prob}, Pred_OS >= {th_os}")

    print("🏗️  Loading 2RR model...", flush=True)
    rr2_model = tf.keras.models.load_model(RR2_MODEL_PATH)

    # ── Load 1RR Fallback Model (baseline) ──────────────────────
    has_baseline = FB_MODEL_PATH.exists() and FB_CONFIG_PATH.exists()
    fb_model = None
    fb_config = None
    if has_baseline:
        fb_config = load_config(FB_CONFIG_PATH)
        print(f"⚙️  1RR Fallback Thresholds: Prob >= {fb_config['Prob_Win_threshold']}, "
              f"OS >= {fb_config['Pred_OS_threshold']}")
        print("🏗️  Loading 1RR fallback model...", flush=True)
        fb_model = tf.keras.models.load_model(FB_MODEL_PATH)

    report = {}

    for split_name in ["test", "holdout"]:
        print("\n" + "=" * 60)
        print(f" 🧪 EVALUATION: {split_name.upper()}")
        print("=" * 60)

        # ── Load 2RR tensors ────────────────────────────────────
        rr2_micro_path = RR2_TENSOR_DIR / f"{split_name}_micro.npy"
        if not rr2_micro_path.exists():
            print(f"⚠️  Skipping {split_name} (no 2RR tensors)")
            continue

        micro = np.load(rr2_micro_path)
        macro = np.load(RR2_TENSOR_DIR / f"{split_name}_macro.npy")
        y_2rr = np.load(RR2_TENSOR_DIR / f"{split_name}_y_class.npy")
        y_mag = np.load(RR2_TENSOR_DIR / f"{split_name}_y_mag.npy")

        if len(micro) == 0:
            print(f"⚠️  Skipping {split_name} (empty)")
            continue

        n_total = len(micro)
        base_2rr_rate = float(np.mean(y_2rr))
        print(f"\n📂 {n_total:,} samples, base 2RR rate: {base_2rr_rate:.1%}")

        # ── 2RR Model predictions ───────────────────────────────
        prob_2rr, pred_os = safe_predict(rr2_model, micro, macro)

        # Sweep thresholds on test set
        print(f"\n📊 2RR Model — Threshold Sweep on {split_name.upper()}:")
        print(f"{'Prob ≥':>8s}  {'OS ≥':>8s}  {'Trades':>8s}  {'2RR WR':>8s}  "
              f"{'EV':>8s}  {'Status':>12s}")
        print("-" * 62)

        best_ev = -999
        best_result = None

        for tp in np.arange(0.30, 0.91, 0.10):
            for to in [0.5, 1.0, 1.5, 2.0]:
                mask = (prob_2rr >= tp) & (pred_os >= to)
                nt = mask.sum()
                if nt < 5:
                    continue
                wr = float(np.mean(y_2rr[mask]))
                ev = 3 * wr - 1
                status = "✅ PROFIT" if ev > 0 else "❌ LOSS"
                print(f"{tp:>8.2f}  {to:>8.1f}  {nt:>8d}  {wr:>8.1%}  "
                      f"{ev:>8.3f}R  {status:>12s}")
                if ev > best_ev and nt >= 10:
                    best_ev = ev
                    best_result = {
                        "th_prob": float(tp), "th_os": float(to),
                        "n_trades": int(nt), "wr_2rr": round(wr, 4),
                        "ev": round(ev, 4),
                        "n_wins": int(np.sum(y_2rr[mask])),
                        "n_losses": int(np.sum(y_2rr[mask] == 0)),
                    }

        # Using config thresholds
        mask_cfg = (prob_2rr >= th_prob) & (pred_os >= th_os)
        n_cfg = mask_cfg.sum()
        if n_cfg > 0:
            wr_cfg = float(np.mean(y_2rr[mask_cfg]))
            ev_cfg = 3 * wr_cfg - 1
            wins_cfg = int(np.sum(y_2rr[mask_cfg]))
            losses_cfg = n_cfg - wins_cfg
        else:
            wr_cfg, ev_cfg, wins_cfg, losses_cfg = 0, -1, 0, 0

        print(f"\n📊 2RR MODEL (config thresholds: {th_prob}, {th_os}):")
        print(f"   Base 2RR rate:  {base_2rr_rate:.1%}")
        print(f"   Filtered 2RR WR: {wr_cfg:.1%} ({n_cfg} trades)")
        print(f"   Wins: {wins_cfg}, Losses: {losses_cfg}")
        print(f"   EV per trade (2:1 RR): {ev_cfg:+.3f}R")
        print(f"   {'✅ PROFITABLE' if ev_cfg > 0 else '❌ UNPROFITABLE'}")

        report[f"2rr_{split_name}"] = {
            "base_2rr_rate": round(base_2rr_rate, 4),
            "config_wr": round(wr_cfg, 4),
            "config_trades": n_cfg,
            "config_ev": round(ev_cfg, 4),
            "config_wins": wins_cfg,
            "config_losses": losses_cfg,
            "best_result": best_result,
        }

        # ── 1RR Fallback Model for comparison ──────────────────
        if has_baseline:
            fb_micro_path = FB_TENSOR_DIR / f"{split_name}_micro.npy"
            if fb_micro_path.exists():
                fb_micro = np.load(fb_micro_path)
                fb_macro = np.load(FB_TENSOR_DIR / f"{split_name}_macro.npy")
                fb_y_mag = np.load(FB_TENSOR_DIR / f"{split_name}_y_mag.npy")

                # Create 2RR labels from 1RR tensors
                fb_y_2rr = (fb_y_mag >= 2.0).astype(float)

                fb_prob, fb_os = safe_predict(fb_model, fb_micro, fb_macro)

                fb_th_p = fb_config["Prob_Win_threshold"]
                fb_th_o = fb_config["Pred_OS_threshold"]
                fb_mask = (fb_prob >= fb_th_p) & (fb_os >= fb_th_o)
                fb_n = fb_mask.sum()

                if fb_n > 0:
                    fb_wr_2rr = float(np.mean(fb_y_2rr[fb_mask]))
                    fb_ev = 3 * fb_wr_2rr - 1
                    fb_wins = int(np.sum(fb_y_2rr[fb_mask]))
                    fb_losses = fb_n - fb_wins
                else:
                    fb_wr_2rr, fb_ev, fb_wins, fb_losses = 0, -1, 0, 0

                print(f"\n📊 1RR FALLBACK MODEL (as 2RR baseline):")
                print(f"   Filtered 2RR WR: {fb_wr_2rr:.1%} ({fb_n} trades)")
                print(f"   Wins: {fb_wins}, Losses: {fb_losses}")
                print(f"   EV per trade (2:1 RR): {fb_ev:+.3f}R")

                if n_cfg > 0 and fb_n > 0:
                    delta_wr = wr_cfg - fb_wr_2rr
                    delta_ev = ev_cfg - fb_ev
                    print(f"\n   📈 DELTA (2RR Model - 1RR Fallback):")
                    print(f"      ΔWR:     {delta_wr:+.1%}")
                    print(f"      ΔEV:     {delta_ev:+.3f}R")
                    print(f"      ΔTrades: {n_cfg - fb_n:+d}")

                report[f"fallback_1rr_{split_name}"] = {
                    "wr_2rr": round(fb_wr_2rr, 4),
                    "trades": fb_n,
                    "ev": round(fb_ev, 4),
                    "wins": fb_wins,
                    "losses": fb_losses,
                }

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Split':<8} {'2RR WR':>8} {'Trades':>8} {'EV':>8} {'Status':>10}")
    print("-" * 72)
    for key in sorted(report.keys()):
        m = report[key]
        parts = key.rsplit("_", 1)
        model_name = parts[0]
        split = parts[1]
        wr = m.get("config_wr", m.get("wr_2rr", 0))
        trades = m.get("config_trades", m.get("trades", 0))
        ev = m.get("config_ev", m.get("ev", 0))
        status = "✅" if ev > 0 else "❌"
        print(f"{model_name:<25} {split:<8} {wr:>8.1%} {trades:>8d} {ev:>8.3f}R {status:>10s}")

    # ── Plot ────────────────────────────────────────────────────
    _plot_evaluation(report)

    # ── Save ────────────────────────────────────────────────────
    report_path = RR2_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))
    print(f"\n💾 Report saved to {report_path}")


def _plot_evaluation(report):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = []
    wrs = []
    trades = []
    evs = []

    for key in sorted(report.keys()):
        m = report[key]
        wr = m.get("config_wr", m.get("wr_2rr", 0))
        tr = m.get("config_trades", m.get("trades", 0))
        ev = m.get("config_ev", m.get("ev", 0))
        models.append(key)
        wrs.append(wr)
        trades.append(tr)
        evs.append(ev)

    colors = ['#e74c3c' if '2rr' in m else '#3498db' for m in models]
    x = np.arange(len(models))

    # WR
    axes[0].bar(x, wrs, color=colors, alpha=0.85)
    axes[0].axhline(y=BREAKEVEN_WR, color='gray', linestyle='--', label=f'Breakeven ({BREAKEVEN_WR:.1%})')
    axes[0].set_ylabel('2RR Win Rate')
    axes[0].set_title('2RR Win Rate')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace('_', '\n') for m in models], fontsize=8)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Trades
    axes[1].bar(x, trades, color=colors, alpha=0.85)
    axes[1].set_ylabel('Number of Trades')
    axes[1].set_title('Trades Taken')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace('_', '\n') for m in models], fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')

    # EV
    bars = axes[2].bar(x, evs, color=['green' if e > 0 else 'red' for e in evs], alpha=0.85)
    axes[2].axhline(y=0, color='gray', linestyle='-')
    axes[2].set_ylabel('EV per Trade (R)')
    axes[2].set_title('Expected Value (2:1 RR)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([m.replace('_', '\n') for m in models], fontsize=8)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('2RR Overshot — Model Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RR2_PLOT_DIR / "evaluation_bar.png", dpi=150)
    plt.close()
    print(f"📊 Evaluation plot saved → {RR2_PLOT_DIR / 'evaluation_bar.png'}")


if __name__ == "__main__":
    print("=" * 60)
    print(" 2RR Overshot: Evaluation & Comparison")
    print("=" * 60)
    evaluate()
