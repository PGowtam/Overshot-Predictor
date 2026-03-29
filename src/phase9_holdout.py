"""
Phase 9.1/9.2 — Holdout (2024) Exec-Priced Pipeline

Generates execution-priced labels + features + buffers + tensors for the
2024 holdout dataset, then evaluates the existing mid-price model on them.
"""

import sys
import json
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

DATA_DIR = BASE_DIR / "Data" / "Raw"
CSV_PATH = DATA_DIR / "renko_with_tick_outcomes_no_be_24_local.csv"
TICK_DIR = DATA_DIR / "Ticks"

EXEC_HOLDOUT_DIR = BASE_DIR / "outputs" / "exec" / "holdout"
EXEC_HOLDOUT_FEATURES = EXEC_HOLDOUT_DIR / "features"
EXEC_HOLDOUT_TENSORS = EXEC_HOLDOUT_DIR / "tensors"

EXEC_DIR = BASE_DIR / "outputs" / "exec"
MAIN_TENSOR_DIR = BASE_DIR / "outputs" / "tensors"
ORIG_MODEL_PATH = BASE_DIR / "outputs" / "model.keras"
ORIG_CONFIG_PATH = BASE_DIR / "outputs" / "config.json"


def step_1_labels():
    print("\n" + "=" * 60)
    print(" HOLDOUT 9.1 — LABEL GENERATION (execution pricing)")
    print("=" * 60, flush=True)
    EXEC_HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    import label_generator as lg

    df = lg.generate_all_labels(
        renko_csv_path=str(CSV_PATH),
        tick_dir=str(TICK_DIR),
        pricing_mode="execution"
    )
    lg.validate_labels(df)

    save_df = df.copy()
    if "sequence" in save_df.columns:
        save_df["sequence"] = save_df["sequence"].astype(str)
    out_path = EXEC_HOLDOUT_DIR / "labels.parquet"
    save_df.to_parquet(out_path, index=False)
    print(f"\n💾 Saved {len(df)} holdout exec labels to {out_path}")

    # Directional WR
    resolved = df[~df["exclude_flag"]]
    long_mask = resolved["uptrend"] == True
    short_mask = resolved["uptrend"] == False
    print(f"\n📊 Holdout Exec Directional WR:")
    print(f"   LONG:    {resolved.loc[long_mask, 'y_class'].mean():.2%} ({long_mask.sum()} bricks)")
    print(f"   SHORT:   {resolved.loc[short_mask, 'y_class'].mean():.2%} ({short_mask.sum()} bricks)")
    print(f"   Overall: {resolved['y_class'].mean():.2%} ({len(resolved)} bricks)")


def step_2_features():
    print("\n" + "=" * 60)
    print(" HOLDOUT 9.1 — FEATURE ENGINEERING")
    print("=" * 60, flush=True)
    EXEC_HOLDOUT_FEATURES.mkdir(parents=True, exist_ok=True)

    import feature_engine as fe
    fe.OUTPUT_DIR = EXEC_HOLDOUT_DIR
    fe.FEATURE_DIR = EXEC_HOLDOUT_FEATURES
    fe.TICK_DIR = TICK_DIR
    fe.main()


def step_3_buffers():
    print("\n" + "=" * 60)
    print(" HOLDOUT 9.1 — BUFFER SIMULATION")
    print("=" * 60, flush=True)

    import buffer_sim as bs
    bs.simulate_buffers(feature_dir=EXEC_HOLDOUT_FEATURES)
    bs.validate_buffers(feature_dir=EXEC_HOLDOUT_FEATURES)


def step_4_tensors():
    print("\n" + "=" * 60)
    print(" HOLDOUT 9.1 — TENSOR CONSTRUCTION")
    print("=" * 60, flush=True)
    EXEC_HOLDOUT_TENSORS.mkdir(parents=True, exist_ok=True)

    import tensor_builder as tb
    tb.OUTPUT_DIR = EXEC_HOLDOUT_DIR
    tb.FEATURE_DIR = EXEC_HOLDOUT_FEATURES
    tb.SNAPSHOT_DIR = EXEC_HOLDOUT_FEATURES / "snapshots"
    tb.TENSOR_DIR = EXEC_HOLDOUT_TENSORS

    # Force all bricks to 'holdout' split
    tb.assign_split = lambda date: "holdout"
    tb.build_and_save_tensors()


def step_5_evaluate():
    print("\n" + "=" * 60)
    print(" HOLDOUT 9.2 — EXISTING MODEL EVALUATION")
    print("=" * 60, flush=True)

    with open(ORIG_CONFIG_PATH) as f:
        config = json.load(f)
    th_p = config["Prob_Win_threshold"]
    th_o = config["Pred_OS_threshold"]
    model = tf.keras.models.load_model(ORIG_MODEL_PATH)

    # Load exec holdout tensors
    exec_micro = np.load(EXEC_HOLDOUT_TENSORS / "holdout_micro.npy")
    exec_macro = np.load(EXEC_HOLDOUT_TENSORS / "holdout_macro.npy")
    exec_yc = np.load(EXEC_HOLDOUT_TENSORS / "holdout_y_class.npy")

    # Load original holdout tensors
    orig_micro = np.load(MAIN_TENSOR_DIR / "holdout_micro.npy")
    orig_macro = np.load(MAIN_TENSOR_DIR / "holdout_macro.npy")
    orig_yc = np.load(MAIN_TENSOR_DIR / "holdout_y_class.npy")

    # Predict on original holdout (baseline)
    n = len(orig_micro)
    pw_orig, po_orig = [], []
    for i in range(0, n, 64):
        end = min(i + 64, n)
        preds = model([orig_micro[i:end], orig_macro[i:end]], training=False)
        pw_orig.append(preds[0].numpy().flatten())
        po_orig.append(preds[1].numpy().flatten())
    pw_orig = np.concatenate(pw_orig)
    po_orig = np.concatenate(po_orig)
    mask_orig = (pw_orig >= th_p) & (po_orig >= th_o)
    n_orig = np.sum(mask_orig)
    wr_orig = np.mean(orig_yc[mask_orig]) if n_orig > 0 else 0.0

    # Score against exec labels
    # Since features are tick-derived (not label-derived), same inputs → same predictions
    # We just judge against exec ground truth
    if len(orig_yc) == len(exec_yc):
        wr_exec = np.mean(exec_yc[mask_orig]) if n_orig > 0 else 0.0
        n_exec = n_orig
    else:
        # Different sample counts → predict on exec tensors
        pw_e, po_e = [], []
        ne = len(exec_micro)
        for i in range(0, ne, 64):
            end = min(i + 64, ne)
            preds = model([exec_micro[i:end], exec_macro[i:end]], training=False)
            pw_e.append(preds[0].numpy().flatten())
            po_e.append(preds[1].numpy().flatten())
        pw_e = np.concatenate(pw_e)
        po_e = np.concatenate(po_e)
        mask_e = (pw_e >= th_p) & (po_e >= th_o)
        n_exec = np.sum(mask_e)
        wr_exec = np.mean(exec_yc[mask_e]) if n_exec > 0 else 0.0

    print(f"\n📊 HOLDOUT Results:")
    print(f"   Mid-Price WR:    {wr_orig:.2%} ({n_orig} trades)")
    print(f"   Exec-Price WR:   {wr_exec:.2%} ({n_exec} trades)")
    print(f"   ΔWR:             {wr_orig - wr_exec:+.2%}")

    # Update the report
    report_path = EXEC_DIR / "phase9_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    else:
        report = {}

    report["holdout"] = {
        "mid_wr": round(float(wr_orig), 4),
        "mid_trades": int(n_orig),
        "exec_wr": round(float(wr_exec), 4),
        "exec_trades": int(n_exec),
        "delta_wr": round(float(wr_orig - wr_exec), 4)
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Updated report → {report_path}")


def main():
    if not CSV_PATH.exists():
        print(f"❌ Input CSV not found: {CSV_PATH}")
        return

    step_1_labels()
    step_2_features()
    step_3_buffers()
    step_4_tensors()
    step_5_evaluate()

    print("\n✅ Holdout exec-priced pipeline complete.")


if __name__ == "__main__":
    main()
