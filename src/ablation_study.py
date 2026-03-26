"""
Phase 8.2: Volume Feature Mitigation — Ablation & Feature Importance (FR-EV-02)

Answers: "How dependent is the model on each of the 9 input features?"

9D Feature Vector:
  [0] z_OFI       — VOLUME-DEPENDENT
  [1] z_Depth     — VOLUME-DEPENDENT
  [2] z_Susc      — VOLUME-DEPENDENT
  [3] z_Vel       — volume-independent
  [4] z_Spread    — volume-independent
  [5] Progress    — volume-independent
  [6] Flag_Curr   — volume-independent
  [7] Flag_Zone   — volume-independent
  [8] Decay       — volume-independent

Steps:
  1. Ablation: Zero out volume features (cols 0-2), retrain, measure WR drop
  2. Tick Direction: Replace col 0 with sign(ΔProgress), retrain, compare
  3. Volume Ratio: SKIPPED (requires full pipeline re-run)
  4. Permutation Importance: Shuffle each feature, measure baseline WR drop

Output:
  outputs/plots/feature_importance.png
  outputs/ablation_report.json
"""

import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from model import build_model, compile_model

OUTPUT_DIR = BASE_DIR / "outputs"
TENSOR_DIR = OUTPUT_DIR / "tensors"
MODEL_PATH = OUTPUT_DIR / "model.keras"
CONFIG_PATH = OUTPUT_DIR / "config.json"
PLOT_DIR = OUTPUT_DIR / "plots"

FEATURE_NAMES = [
    "z_OFI", "z_Depth", "z_Susc",  # Volume-dependent (0-2)
    "z_Vel", "z_Spread", "Progress",  # Volume-independent (3-5)
    "Flag_Curr", "Flag_Zone", "Decay"  # Volume-independent (6-8)
]
VOLUME_COLS = [0, 1, 2]

# ── Helpers ────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def load_split(split):
    micro = np.load(TENSOR_DIR / f"{split}_micro.npy")
    macro = np.load(TENSOR_DIR / f"{split}_macro.npy")
    y_class = np.load(TENSOR_DIR / f"{split}_y_class.npy")
    y_mag = np.load(TENSOR_DIR / f"{split}_y_mag.npy")
    return micro, macro, y_class, y_mag

def safe_predict(model, micro, macro, batch_size=64):
    """Predict in batches to avoid memory issues."""
    n = len(micro)
    prob_wins, pred_oss = [], []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        preds = model([micro[i:end], macro[i:end]], training=False)
        prob_wins.append(preds[0].numpy().flatten())
        pred_oss.append(preds[1].numpy().flatten())
    return np.concatenate(prob_wins), np.concatenate(pred_oss)

def compute_wr(model, micro, macro, y_class, th_p, th_o):
    """Compute model-filtered win rate."""
    prob_win, pred_os = safe_predict(model, micro, macro)
    mask = (prob_win >= th_p) & (pred_os >= th_o)
    n_trades = np.sum(mask)
    if n_trades == 0:
        return 0.0, 0
    wr = np.mean(y_class[mask])
    return float(wr), int(n_trades)

def train_model_on_data(train_micro, train_macro, train_y_class, train_y_mag,
                        val_micro, val_macro, val_y_class, val_y_mag,
                        train_weights=None, max_epochs=100, patience=15):
    """Train a fresh model on given data. Returns the trained model."""
    model = build_model()
    model = compile_model(model)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience,
            restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=8, verbose=0, min_lr=1e-6
        ),
    ]

    sw = None
    if train_weights is not None:
        sw = [train_weights, train_weights]

    model.fit(
        x=[train_micro, train_macro],
        y=[train_y_class, train_y_mag],
        validation_data=(
            [val_micro, val_macro],
            [val_y_class, val_y_mag]
        ),
        sample_weight=sw,
        epochs=max_epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=0
    )
    return model


# ═══════════════════════════════════════════════════════════════
# STEP 1: Ablation — Zero Volume Features
# ═══════════════════════════════════════════════════════════════

def step_1_ablation(config, results):
    print("\n" + "=" * 60)
    print(" STEP 1: ABLATION — Zero Volume Features (cols 0-2)")
    print("=" * 60, flush=True)

    th_p, th_o = config["Prob_Win_threshold"], config["Pred_OS_threshold"]

    # Load data
    tr_micro, tr_macro, tr_yc, tr_ym = load_split("train")
    va_micro, va_macro, va_yc, va_ym = load_split("val")
    te_micro, te_macro, te_yc, te_ym = load_split("test")
    ho_micro, ho_macro, ho_yc, ho_ym = load_split("holdout")

    tr_weights_path = TENSOR_DIR / "train_weights.npy"
    tr_weights = np.load(tr_weights_path) if tr_weights_path.exists() else None

    # Zero out volume columns
    def zero_volume(micro):
        m = micro.copy()
        m[:, :, :, VOLUME_COLS] = 0.0
        return m

    tr_micro_abl = zero_volume(tr_micro)
    va_micro_abl = zero_volume(va_micro)
    te_micro_abl = zero_volume(te_micro)
    ho_micro_abl = zero_volume(ho_micro)

    print("🔧 Training ablated model (6 features, volume zeroed)...", flush=True)
    abl_model = train_model_on_data(
        tr_micro_abl, tr_macro, tr_yc, tr_ym,
        va_micro_abl, va_macro, va_yc, va_ym,
        train_weights=tr_weights
    )

    # Evaluate
    te_wr, te_n = compute_wr(abl_model, te_micro_abl, te_macro, te_yc, th_p, th_o)
    ho_wr, ho_n = compute_wr(abl_model, ho_micro_abl, ho_macro, ho_yc, th_p, th_o)

    print(f"\n📊 Ablated Model (No Volume):")
    print(f"   Test WR:    {te_wr:.2%} ({te_n} trades)")
    print(f"   Holdout WR: {ho_wr:.2%} ({ho_n} trades)")

    results["step_1_ablation"] = {
        "test_wr": round(te_wr, 4), "test_trades": te_n,
        "holdout_wr": round(ho_wr, 4), "holdout_trades": ho_n,
        "zeroed_features": [FEATURE_NAMES[c] for c in VOLUME_COLS]
    }

    # Cleanup
    del tr_micro_abl, va_micro_abl, te_micro_abl, ho_micro_abl, abl_model
    tf.keras.backend.clear_session()

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 2: Tick Direction Encoding
# ═══════════════════════════════════════════════════════════════

def step_2_tick_direction(config, results):
    print("\n" + "=" * 60)
    print(" STEP 2: TICK DIRECTION ENCODING (replace z_OFI)")
    print("=" * 60, flush=True)

    th_p, th_o = config["Prob_Win_threshold"], config["Pred_OS_threshold"]

    tr_micro, tr_macro, tr_yc, tr_ym = load_split("train")
    va_micro, va_macro, va_yc, va_ym = load_split("val")
    te_micro, te_macro, te_yc, te_ym = load_split("test")
    ho_micro, ho_macro, ho_yc, ho_ym = load_split("holdout")

    tr_weights_path = TENSOR_DIR / "train_weights.npy"
    tr_weights = np.load(tr_weights_path) if tr_weights_path.exists() else None

    def replace_ofi_with_tick_dir(micro):
        """Replace col 0 (z_OFI) with sign(ΔProgress).
        
        Progress (col 5) tracks mid-price movement relative to brick.
        sign(diff(Progress)) approximates tick direction.
        """
        m = micro.copy()
        # For each sample and each brick-step
        # Progress is col 5 in the (100, 9) snapshot
        progress = m[:, :, :, 5]  # (N, 10, 100)
        
        # Compute tick-to-tick direction
        tick_dir = np.zeros_like(progress)
        tick_dir[:, :, 1:] = np.sign(np.diff(progress, axis=2))  # (N, 10, 99)
        
        # Replace col 0
        m[:, :, :, 0] = tick_dir
        return m

    tr_micro_td = replace_ofi_with_tick_dir(tr_micro)
    va_micro_td = replace_ofi_with_tick_dir(va_micro)
    te_micro_td = replace_ofi_with_tick_dir(te_micro)
    ho_micro_td = replace_ofi_with_tick_dir(ho_micro)

    print("🔧 Training tick-direction model...", flush=True)
    td_model = train_model_on_data(
        tr_micro_td, tr_macro, tr_yc, tr_ym,
        va_micro_td, va_macro, va_yc, va_ym,
        train_weights=tr_weights
    )

    te_wr, te_n = compute_wr(td_model, te_micro_td, te_macro, te_yc, th_p, th_o)
    ho_wr, ho_n = compute_wr(td_model, ho_micro_td, ho_macro, ho_yc, th_p, th_o)

    print(f"\n📊 Tick Direction Model:")
    print(f"   Test WR:    {te_wr:.2%} ({te_n} trades)")
    print(f"   Holdout WR: {ho_wr:.2%} ({ho_n} trades)")

    results["step_2_tick_direction"] = {
        "test_wr": round(te_wr, 4), "test_trades": te_n,
        "holdout_wr": round(ho_wr, 4), "holdout_trades": ho_n,
        "replaced": "z_OFI -> sign(ΔProgress)"
    }

    del tr_micro_td, va_micro_td, te_micro_td, ho_micro_td, td_model
    tf.keras.backend.clear_session()

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 4: Permutation Feature Importance
# ═══════════════════════════════════════════════════════════════

def step_4_feature_importance(config, results):
    print("\n" + "=" * 60)
    print(" STEP 4: PERMUTATION FEATURE IMPORTANCE")
    print("=" * 60, flush=True)

    th_p, th_o = config["Prob_Win_threshold"], config["Pred_OS_threshold"]

    # Load baseline model
    model = tf.keras.models.load_model(MODEL_PATH)
    te_micro, te_macro, te_yc, te_ym = load_split("test")

    # Baseline WR
    base_wr, base_n = compute_wr(model, te_micro, te_macro, te_yc, th_p, th_o)
    print(f"   Baseline Test WR: {base_wr:.2%} ({base_n} trades)")

    importances = {}
    N_PERMUTATIONS = 5  # Average over multiple shuffles for stability

    for col in range(9):
        wr_drops = []
        for seed in range(N_PERMUTATIONS):
            rng = np.random.RandomState(42 + seed)
            micro_perm = te_micro.copy()
            # Shuffle this feature column across all samples
            # Shape: (N, 10, 100, 9) — shuffle along axis 0
            perm_idx = rng.permutation(len(micro_perm))
            micro_perm[:, :, :, col] = te_micro[perm_idx, :, :, col]

            perm_wr, perm_n = compute_wr(model, micro_perm, te_macro, te_yc, th_p, th_o)
            wr_drops.append(base_wr - perm_wr)

        mean_drop = np.mean(wr_drops)
        importances[FEATURE_NAMES[col]] = round(float(mean_drop), 4)
        tag = "🔴 VOL" if col in VOLUME_COLS else "🟢"
        print(f"   {tag} {FEATURE_NAMES[col]:12s}: ΔWR = {mean_drop:+.2%}")

    # Sort by importance
    sorted_imp = dict(sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(sorted_imp.keys())
    values = list(sorted_imp.values())
    colors = ['#e74c3c' if n in ["z_OFI", "z_Depth", "z_Susc"] else '#2ecc71' for n in names]

    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Win Rate Drop (ΔWR) when feature shuffled", fontsize=12)
    ax.set_title("Permutation Feature Importance (Test Set)", fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:+.2%}', va='center', ha='left', fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Volume-Dependent'),
        Patch(facecolor='#2ecc71', label='Volume-Independent')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plot_path = PLOT_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n📊 Saved feature importance plot → {plot_path}")

    results["step_4_importance"] = {
        "baseline_wr": round(base_wr, 4),
        "baseline_trades": base_n,
        "importances": sorted_imp,
        "n_permutations": N_PERMUTATIONS
    }

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Phase 8.2: Volume Feature Mitigation Workflow")
    print("=" * 60, flush=True)

    config = load_config()
    print(f"⚙️  Thresholds: Prob_Win >= {config['Prob_Win_threshold']}, "
          f"Pred_OS >= {config['Pred_OS_threshold']}")

    # Baseline (existing model)
    print("\n📌 BASELINE (Original 9-feature model):")
    baseline_model = tf.keras.models.load_model(MODEL_PATH)

    te_micro, te_macro, te_yc, _ = load_split("test")
    ho_micro, ho_macro, ho_yc, _ = load_split("holdout")

    th_p = config["Prob_Win_threshold"]
    th_o = config["Pred_OS_threshold"]

    base_te_wr, base_te_n = compute_wr(baseline_model, te_micro, te_macro, te_yc, th_p, th_o)
    base_ho_wr, base_ho_n = compute_wr(baseline_model, ho_micro, ho_macro, ho_yc, th_p, th_o)
    print(f"   Test WR:    {base_te_wr:.2%} ({base_te_n} trades)")
    print(f"   Holdout WR: {base_ho_wr:.2%} ({base_ho_n} trades)")

    del baseline_model
    tf.keras.backend.clear_session()

    results = {
        "baseline": {
            "test_wr": round(base_te_wr, 4), "test_trades": base_te_n,
            "holdout_wr": round(base_ho_wr, 4), "holdout_trades": base_ho_n,
        }
    }

    # Run Steps
    results = step_1_ablation(config, results)
    results = step_2_tick_direction(config, results)
    results["step_3_volume_ratio"] = "SKIPPED (requires full pipeline re-run)"
    results = step_4_feature_importance(config, results)

    # ── Summary Table ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)

    bl = results["baseline"]
    s1 = results["step_1_ablation"]
    s2 = results["step_2_tick_direction"]

    print(f"{'Model':<25} {'Test WR':>10} {'Trades':>8}  {'Holdout WR':>12} {'Trades':>8}")
    print("-" * 70)
    print(f"{'Baseline (9 features)':<25} {bl['test_wr']:>10.2%} {bl['test_trades']:>8d}  {bl['holdout_wr']:>12.2%} {bl['holdout_trades']:>8d}")
    print(f"{'Ablated (6 features)':<25} {s1['test_wr']:>10.2%} {s1['test_trades']:>8d}  {s1['holdout_wr']:>12.2%} {s1['holdout_trades']:>8d}")
    print(f"{'Tick Direction':<25} {s2['test_wr']:>10.2%} {s2['test_trades']:>8d}  {s2['holdout_wr']:>12.2%} {s2['holdout_trades']:>8d}")

    delta_abl_te = bl["test_wr"] - s1["test_wr"]
    delta_abl_ho = bl["holdout_wr"] - s1["holdout_wr"]
    print(f"\n   Volume Feature Impact (ΔWR from ablation):")
    print(f"     Test:    {delta_abl_te:+.2%}")
    print(f"     Holdout: {delta_abl_ho:+.2%}")

    if abs(delta_abl_te) < 0.01:
        print("   → Volume adds <1% WR. Model is NOT volume-dependent. ✅")
    else:
        print(f"   → Volume adds {abs(delta_abl_te):.1%} WR. Model IS volume-dependent. ⚠️")

    # Save
    report_path = OUTPUT_DIR / "ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Report saved → {report_path}")


if __name__ == "__main__":
    main()
