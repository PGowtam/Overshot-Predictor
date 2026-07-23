"""
Volume Fallback: Evaluation & Comparison
==========================================
Evaluates the fallback model on test and holdout sets, and compares
performance against the original execution-priced model.

Reads from:
  - outputs/fallback/{model.keras, config.json, tensors/}
  - outputs/exec/{model.keras, config.json}  (baseline comparison)
  - outputs/tensors/  (original tensors for baseline eval)

Outputs:
  - Console report with side-by-side comparison
  - outputs/fallback/evaluation_report.json
  - outputs/fallback/plots/comparison_bar.png
"""

import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

OUTPUT_DIR = BASE_DIR / "outputs"
FALLBACK_DIR = OUTPUT_DIR / "fallback"

# Fallback paths
FB_TENSOR_DIR = FALLBACK_DIR / "tensors"
FB_MODEL_PATH = FALLBACK_DIR / "model.keras"
FB_CONFIG_PATH = FALLBACK_DIR / "config.json"
FB_PLOT_DIR = FALLBACK_DIR / "plots"

# Exec baseline paths
EXEC_MODEL_PATH = OUTPUT_DIR / "exec" / "model.keras"
EXEC_CONFIG_PATH = OUTPUT_DIR / "exec" / "config.json"
EXEC_TENSOR_DIR = OUTPUT_DIR / "tensors"  # Original tensors live here

FB_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


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


def eval_model(model, micro, macro, y_class, y_mag, th_prob, th_os, label=""):
    """Evaluate a model and return metrics dict."""
    prob_win, pred_os = safe_predict(model, micro, macro)

    # Unfiltered WR
    baseline_wr = float(np.mean(y_class))

    # Model-Filtered WR
    mask = (prob_win >= th_prob) & (pred_os >= th_os)
    n_trades = int(np.sum(mask))

    if n_trades > 0:
        filtered_wr = float(np.mean(y_class[mask]))
        tp = int(np.sum(y_class[mask] == 1))
        fp = int(np.sum(y_class[mask] == 0))
    else:
        filtered_wr = 0.0
        tp, fp = 0, 0

    # Pearson r on WIN samples
    win_mask = (y_class == 1)
    if np.sum(win_mask) > 1:
        corr = float(np.corrcoef(y_mag[win_mask], pred_os[win_mask])[0, 1])
    else:
        corr = 0.0

    return {
        "baseline_wr": round(baseline_wr, 4),
        "filtered_wr": round(filtered_wr, 4),
        "n_trades": n_trades,
        "tp": tp,
        "fp": fp,
        "pearson_r": round(corr, 4),
        "th_prob": th_prob,
        "th_os": th_os,
    }


def evaluate():
    # ── Load Fallback Model & Config ────────────────────────────
    if not FB_MODEL_PATH.exists():
        print("❌ Fallback model not found.")
        return
    if not FB_CONFIG_PATH.exists():
        print("❌ Fallback config not found.")
        return

    fb_config = load_config(FB_CONFIG_PATH)
    fb_th_prob = fb_config["Prob_Win_threshold"]
    fb_th_os = fb_config["Pred_OS_threshold"]

    print(f"⚙️  Fallback Thresholds: Prob_Win >= {fb_th_prob}, Pred_OS >= {fb_th_os}")

    print("🏗️  Loading fallback model...", flush=True)
    fb_model = tf.keras.models.load_model(FB_MODEL_PATH)

    # ── Load Exec Baseline Model & Config ───────────────────────
    has_baseline = EXEC_MODEL_PATH.exists() and EXEC_CONFIG_PATH.exists()
    exec_model = None
    exec_config = None

    if has_baseline:
        exec_config = load_config(EXEC_CONFIG_PATH)
        print(f"⚙️  Exec Thresholds: Prob_Win >= {exec_config['Prob_Win_threshold']}, "
              f"Pred_OS >= {exec_config['Pred_OS_threshold']}")
        print("🏗️  Loading exec baseline model...", flush=True)
        exec_model = tf.keras.models.load_model(EXEC_MODEL_PATH)
    else:
        print("⚠️  Exec baseline model not found. Skipping comparison.")

    # ── Evaluate on Fallback Tensors ────────────────────────────
    report = {}
    for split_name in ["test", "holdout"]:
        print("\n" + "=" * 60)
        print(f" 🧪 EVALUATION: {split_name.upper()}")
        print("=" * 60)

        # Load fallback tensors
        fb_micro_path = FB_TENSOR_DIR / f"{split_name}_micro.npy"
        if not fb_micro_path.exists():
            print(f"⚠️  Skipping {split_name} (no fallback tensors)")
            continue

        fb_micro = np.load(fb_micro_path)
        fb_macro = np.load(FB_TENSOR_DIR / f"{split_name}_macro.npy")
        fb_y_class = np.load(FB_TENSOR_DIR / f"{split_name}_y_class.npy")
        fb_y_mag = np.load(FB_TENSOR_DIR / f"{split_name}_y_mag.npy")

        if len(fb_micro) == 0:
            print(f"⚠️  Skipping {split_name} (empty)")
            continue

        print(f"\n📂 Loaded {len(fb_micro):,} fallback samples")

        # Evaluate fallback model
        fb_metrics = eval_model(fb_model, fb_micro, fb_macro, fb_y_class, fb_y_mag,
                                fb_th_prob, fb_th_os, label="Fallback")

        print(f"\n📊 FALLBACK MODEL:")
        print(f"   Baseline WR (Unfiltered): {fb_metrics['baseline_wr']:.2%}")
        print(f"   Filtered WR:              {fb_metrics['filtered_wr']:.2%} ({fb_metrics['n_trades']} trades)")
        print(f"   Trades: {fb_metrics['tp']} Wins, {fb_metrics['fp']} Losses")
        print(f"   Pearson r (Pred_OS on WINs): {fb_metrics['pearson_r']:.4f}")

        report[f"fallback_{split_name}"] = fb_metrics

        # Evaluate exec baseline on ITS OWN tensors
        if has_baseline:
            exec_micro_path = EXEC_TENSOR_DIR / f"{split_name}_micro.npy"
            if exec_micro_path.exists():
                exec_micro = np.load(exec_micro_path)
                exec_macro = np.load(EXEC_TENSOR_DIR / f"{split_name}_macro.npy")
                exec_y_class = np.load(EXEC_TENSOR_DIR / f"{split_name}_y_class.npy")
                exec_y_mag = np.load(EXEC_TENSOR_DIR / f"{split_name}_y_mag.npy")

                if len(exec_micro) > 0:
                    exec_metrics = eval_model(
                        exec_model, exec_micro, exec_macro, exec_y_class, exec_y_mag,
                        exec_config["Prob_Win_threshold"], exec_config["Pred_OS_threshold"],
                        label="Exec"
                    )

                    print(f"\n📊 EXEC BASELINE:")
                    print(f"   Baseline WR (Unfiltered): {exec_metrics['baseline_wr']:.2%}")
                    print(f"   Filtered WR:              {exec_metrics['filtered_wr']:.2%} ({exec_metrics['n_trades']} trades)")
                    print(f"   Trades: {exec_metrics['tp']} Wins, {exec_metrics['fp']} Losses")
                    print(f"   Pearson r (Pred_OS on WINs): {exec_metrics['pearson_r']:.4f}")

                    report[f"exec_{split_name}"] = exec_metrics

                    # Delta
                    delta_wr = fb_metrics["filtered_wr"] - exec_metrics["filtered_wr"]
                    delta_trades = fb_metrics["n_trades"] - exec_metrics["n_trades"]
                    print(f"\n   📈 DELTA (Fallback - Exec):")
                    print(f"      ΔWR:     {delta_wr:+.2%}")
                    print(f"      ΔTrades: {delta_trades:+d}")
            else:
                print(f"\n⚠️  Exec {split_name} tensors not found for comparison.")

    # ── Summary Table ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'Split':<10} {'Filtered WR':>12} {'Trades':>8} {'Pearson r':>10}")
    print("-" * 70)

    for key in sorted(report.keys()):
        m = report[key]
        model_name, split = key.rsplit("_", 1)
        print(f"{model_name:<25} {split:<10} {m['filtered_wr']:>12.2%} {m['n_trades']:>8d} {m['pearson_r']:>10.4f}")

    # ── Comparison Bar Chart ────────────────────────────────────
    _plot_comparison(report)

    # ── Save Report ─────────────────────────────────────────────
    report_path = FALLBACK_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved to {report_path}")


def _plot_comparison(report):
    """Generate a comparison bar chart."""
    # Collect data for plotting
    splits = ["test", "holdout"]
    models = ["fallback", "exec"]
    colors = {"fallback": "#e67e22", "exec": "#3498db"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, metric in enumerate(["filtered_wr", "n_trades"]):
        ax = axes[ax_idx]
        x = np.arange(len(splits))
        width = 0.35

        for m_idx, model in enumerate(models):
            values = []
            for split in splits:
                key = f"{model}_{split}"
                if key in report:
                    values.append(report[key][metric])
                else:
                    values.append(0)

            offset = (m_idx - 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model.title(), color=colors[model], alpha=0.85)

            # Value labels
            for bar, val in zip(bars, values):
                if metric == "filtered_wr":
                    label = f"{val:.1%}"
                else:
                    label = f"{val}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        label, ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([s.title() for s in splits])
        if metric == "filtered_wr":
            ax.set_ylabel("Win Rate")
            ax.set_title("Model-Filtered Win Rate")
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylabel("Number of Trades")
            ax.set_title("Trades Taken")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Volume Fallback vs Exec Baseline", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = FB_PLOT_DIR / "comparison_bar.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Comparison plot saved → {plot_path}")


if __name__ == "__main__":
    print("=" * 60)
    print(" Volume Fallback: Evaluation & Comparison")
    print("=" * 60)
    evaluate()
