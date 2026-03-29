"""
Phase 9.1 & 9.2: Market Realism Recalibration

Pipeline:
  9.1 — Generate execution-priced labels, run features → buffers → tensors
  9.2 — Evaluate EXISTING mid-price model on execution-priced tensors

Execution pricing:
  LONG exits → scan with bid (what you receive when selling)
  SHORT exits → scan with ask (what you pay when buying back)

Usage:
  python src/phase9_pipeline.py
"""

import sys
import json
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

DATA_DIR = BASE_DIR / "Data" / "Raw"
CSV_PATH = DATA_DIR / "renko_with_tick_outcomes_no_be_XAUUSD20-24.csv"
TICK_DIR = DATA_DIR / "Ticks"

# Execution-priced outputs go in outputs/exec/
EXEC_DIR = BASE_DIR / "outputs" / "exec"
EXEC_FEATURES = EXEC_DIR / "features"
EXEC_TENSORS = EXEC_DIR / "tensors"

# Original model + config
ORIG_MODEL_PATH = BASE_DIR / "outputs" / "model.keras"
ORIG_CONFIG_PATH = BASE_DIR / "outputs" / "config.json"
ORIG_TENSOR_DIR = BASE_DIR / "outputs" / "tensors"

PRICING_MODE = "execution"


# ═══════════════════════════════════════════════════════════════
# STEP 1: Label Generation (Execution Pricing)
# ═══════════════════════════════════════════════════════════════

def step_1_labels():
    print("\n" + "=" * 60)
    print(f" 9.1a — LABEL GENERATION (pricing_mode='{PRICING_MODE}')")
    print("=" * 60, flush=True)
    EXEC_DIR.mkdir(parents=True, exist_ok=True)

    import label_generator as lg

    df = lg.generate_all_labels(
        renko_csv_path=str(CSV_PATH),
        tick_dir=str(TICK_DIR),
        pricing_mode=PRICING_MODE
    )

    # Run validation
    lg.validate_labels(df)

    # Save
    save_df = df.copy()
    if "sequence" in save_df.columns:
        save_df["sequence"] = save_df["sequence"].astype(str)
    out_path = EXEC_DIR / "labels.parquet"
    save_df.to_parquet(out_path, index=False)
    print(f"\n💾 Saved {len(df)} exec-priced labels to {out_path}", flush=True)

    # ── Directional WR comparison ──────────────────────────────
    resolved = df[~df["exclude_flag"]]
    long_mask = resolved["uptrend"] == True
    short_mask = resolved["uptrend"] == False

    long_wr = resolved.loc[long_mask, "y_class"].mean()
    short_wr = resolved.loc[short_mask, "y_class"].mean()
    overall_wr = resolved["y_class"].mean()

    print(f"\n📊 Directional Win Rates (Execution Pricing):")
    print(f"   LONG:    {long_wr:.2%} ({long_mask.sum():,} bricks)")
    print(f"   SHORT:   {short_wr:.2%} ({short_mask.sum():,} bricks)")
    print(f"   Overall: {overall_wr:.2%} ({len(resolved):,} bricks)")

    # Load mid-price labels for comparison
    mid_labels = pd.read_parquet(BASE_DIR / "outputs" / "labels.parquet")
    mid_resolved = mid_labels[~mid_labels["exclude_flag"]]
    mid_long = mid_resolved[mid_resolved["uptrend"] == True]
    mid_short = mid_resolved[mid_resolved["uptrend"] == False]

    print(f"\n📊 Directional Win Rates (Mid-Price — Baseline):")
    print(f"   LONG:    {mid_long['y_class'].mean():.2%} ({len(mid_long):,} bricks)")
    print(f"   SHORT:   {mid_short['y_class'].mean():.2%} ({len(mid_short):,} bricks)")
    print(f"   Overall: {mid_resolved['y_class'].mean():.2%} ({len(mid_resolved):,} bricks)")

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2: Feature Engineering
# ═══════════════════════════════════════════════════════════════

def step_2_features():
    print("\n" + "=" * 60)
    print(" 9.1b — FEATURE ENGINEERING (Execution Labels)")
    print("=" * 60, flush=True)
    EXEC_FEATURES.mkdir(parents=True, exist_ok=True)

    import feature_engine as fe

    # Monkey-patch paths to use exec directory
    fe.OUTPUT_DIR = EXEC_DIR
    fe.FEATURE_DIR = EXEC_FEATURES
    fe.TICK_DIR = TICK_DIR

    fe.main()


# ═══════════════════════════════════════════════════════════════
# STEP 3: Buffer Simulation
# ═══════════════════════════════════════════════════════════════

def step_3_buffers():
    print("\n" + "=" * 60)
    print(" 9.1c — BUFFER SIMULATION (Execution Labels)")
    print("=" * 60, flush=True)

    import buffer_sim as bs

    bs.simulate_buffers(feature_dir=EXEC_FEATURES)
    bs.validate_buffers(feature_dir=EXEC_FEATURES)


# ═══════════════════════════════════════════════════════════════
# STEP 4: Tensor Construction
# ═══════════════════════════════════════════════════════════════

def step_4_tensors():
    print("\n" + "=" * 60)
    print(" 9.1d — TENSOR CONSTRUCTION (Execution Labels)")
    print("=" * 60, flush=True)
    EXEC_TENSORS.mkdir(parents=True, exist_ok=True)

    import tensor_builder as tb

    # Monkey-patch paths
    tb.OUTPUT_DIR = EXEC_DIR
    tb.FEATURE_DIR = EXEC_FEATURES
    tb.SNAPSHOT_DIR = EXEC_FEATURES / "snapshots"
    tb.TENSOR_DIR = EXEC_TENSORS

    # Keep standard split assignment (train/val/test/holdout by date)
    tb.build_and_save_tensors()


# ═══════════════════════════════════════════════════════════════
# STEP 5: Evaluate Existing Model (Phase 9.2)
# ═══════════════════════════════════════════════════════════════

def safe_predict(model, micro, macro, batch_size=64):
    """Predict in batches."""
    n = len(micro)
    prob_wins, pred_oss = [], []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        preds = model([micro[i:end], macro[i:end]], training=False)
        prob_wins.append(preds[0].numpy().flatten())
        pred_oss.append(preds[1].numpy().flatten())
    return np.concatenate(prob_wins), np.concatenate(pred_oss)


def step_5_evaluate():
    print("\n" + "=" * 60)
    print(" 9.2 — EXISTING MODEL ON EXECUTION-PRICED TENSORS")
    print("=" * 60, flush=True)

    # Load config & model
    with open(ORIG_CONFIG_PATH) as f:
        config = json.load(f)
    th_p = config["Prob_Win_threshold"]
    th_o = config["Pred_OS_threshold"]

    print(f"⚙️  Thresholds: Prob_Win >= {th_p}, Pred_OS >= {th_o}")
    model = tf.keras.models.load_model(ORIG_MODEL_PATH)

    # Load execution-priced labels for directional breakdown
    exec_labels = pd.read_parquet(EXEC_DIR / "labels.parquet")

    results = {}

    for split in ["test", "holdout"]:
        print(f"\n{'─' * 40}")
        print(f" {split.upper()}")
        print(f"{'─' * 40}")

        # ── Load exec-priced tensors ──────────────────────────
        exec_micro_path = EXEC_TENSORS / f"{split}_micro.npy"
        exec_macro_path = EXEC_TENSORS / f"{split}_macro.npy"
        exec_yc_path = EXEC_TENSORS / f"{split}_y_class.npy"

        if not exec_micro_path.exists():
            print(f"   ⚠️  {split} exec tensors not found. Skipping.")
            continue

        exec_micro = np.load(exec_micro_path)
        exec_macro = np.load(exec_macro_path)
        exec_yc = np.load(exec_yc_path)

        # ── Original mid-price tensors for comparison ─────────
        orig_micro = np.load(ORIG_TENSOR_DIR / f"{split}_micro.npy")
        orig_macro = np.load(ORIG_TENSOR_DIR / f"{split}_macro.npy")
        orig_yc = np.load(ORIG_TENSOR_DIR / f"{split}_y_class.npy")

        # ── Predict on ORIGINAL tensors (baseline) ────────────
        prob_win_orig, pred_os_orig = safe_predict(model, orig_micro, orig_macro)
        mask_orig = (prob_win_orig >= th_p) & (pred_os_orig >= th_o)
        n_orig = np.sum(mask_orig)
        wr_orig = np.mean(orig_yc[mask_orig]) if n_orig > 0 else 0.0

        # ── Predict on EXEC tensors with ORIGINAL model ───────
        # Note: Same model, but exec tensors have different y_class labels
        # The model's predictions (prob_win, pred_os) depend on the INPUT features
        # which are recomputed from exec labels. But the features themselves
        # (OFI, Depth, etc.) are computed from TICKS, not from labels.
        # So the model inputs are the SAME, but the ground truth changes.
        #
        # IMPORTANT: Features are tick-derived, labels don't affect features.
        # So we can use the ORIGINAL model's predictions on ORIGINAL inputs
        # but score them against EXECUTION-PRICED ground truth.

        # Use original model predictions, but benchmark against exec y_class
        # We need to handle the case where exec and orig have different sample counts

        min_n = min(len(orig_yc), len(exec_yc))

        if len(orig_yc) == len(exec_yc):
            # Same samples — direct comparison
            wr_exec = np.mean(exec_yc[mask_orig]) if n_orig > 0 else 0.0
            n_exec = n_orig
        else:
            # Different sample counts (some bricks resolved differently)
            # Score the model on exec tensors directly
            prob_win_exec, pred_os_exec = safe_predict(model, exec_micro, exec_macro)
            mask_exec = (prob_win_exec >= th_p) & (pred_os_exec >= th_o)
            n_exec = np.sum(mask_exec)
            wr_exec = np.mean(exec_yc[mask_exec]) if n_exec > 0 else 0.0

        print(f"   Mid-Price WR:    {wr_orig:.2%} ({n_orig} trades)")
        print(f"   Exec-Price WR:   {wr_exec:.2%} ({n_exec} trades)")
        print(f"   ΔWR:             {wr_orig - wr_exec:+.2%}")

        results[split] = {
            "mid_wr": round(float(wr_orig), 4),
            "mid_trades": int(n_orig),
            "exec_wr": round(float(wr_exec), 4),
            "exec_trades": int(n_exec),
            "delta_wr": round(float(wr_orig - wr_exec), 4)
        }

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" PHASE 9.2 SUMMARY — Existing Model Benchmark")
    print("=" * 60)
    print(f"{'Split':<12} {'Mid-Price WR':>14} {'Trades':>8}  {'Exec-Price WR':>15} {'Trades':>8}  {'ΔWR':>8}")
    print("-" * 70)
    for split in ["test", "holdout"]:
        if split in results:
            r = results[split]
            print(f"{split.upper():<12} {r['mid_wr']:>14.2%} {r['mid_trades']:>8d}  "
                  f"{r['exec_wr']:>15.2%} {r['exec_trades']:>8d}  {r['delta_wr']:>+8.2%}")

    # Save results
    report_path = EXEC_DIR / "phase9_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Report saved → {report_path}")

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Phase 9: Market Realism Recalibration")
    print(f" Pricing Mode: {PRICING_MODE}")
    print("=" * 60, flush=True)

    if not CSV_PATH.exists():
        print(f"❌ Input CSV not found: {CSV_PATH}")
        return

    # 9.1: Generate execution-priced labels + features + buffers + tensors
    step_1_labels()
    step_2_features()
    step_3_buffers()
    step_4_tensors()

    # 9.2: Evaluate existing model on exec-priced tensors
    step_5_evaluate()

    print("\n✅ Phase 9.1 & 9.2 complete.")


if __name__ == "__main__":
    main()
