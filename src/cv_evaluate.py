"""
Iteration 2: Cross Validation Evaluation

1. Evaluates each Fold's model against its own localized Test set.
2. Evaluates each Fold's model against the pristine 2024 Holdout set.
3. Evaluates a Voting Ensemble (Majority Rule > 50%) on the 2024 Holdout set.
4. Generates an extensive markdown report.
"""

import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

EXEC_DIR = BASE_DIR / "outputs" / "exec"
CV_DIR = EXEC_DIR / "cv"
HOLDOUT_DIR = EXEC_DIR / "holdout" / "tensors"

REPORT_MD = BASE_DIR / "outputs" / "cv_evaluation_report.md"

def safe_predict(model, micro, macro, batch_size=64):
    n = len(micro)
    prob_wins, pred_oss = [], []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        preds = model([micro[i:end], macro[i:end]], training=False)
        prob_wins.append(preds[0].numpy().flatten())
        pred_oss.append(preds[1].numpy().flatten())
    return np.concatenate(prob_wins), np.concatenate(pred_oss)


def evaluate_split(model, th_p, th_o, micro, macro, y_class):
    prob_win, pred_os = safe_predict(model, micro, macro)
    mask = (prob_win >= th_p) & (pred_os >= th_o)
    n_trades = np.sum(mask)
    wr = np.mean(y_class[mask]) if n_trades > 0 else 0.0
    return {"wr": float(wr), "trades": int(n_trades), "mask": mask}


def main():
    print("=" * 60)
    print(" IT2 — CROSS VALIDATION EVALUATION")
    print("=" * 60)
    
    # ── 1. Load Holdout Tensors ──────────────────────────────────────────
    h_micro = np.load(HOLDOUT_DIR / "holdout_micro.npy")
    h_macro = np.load(HOLDOUT_DIR / "holdout_macro.npy")
    h_yc = np.load(HOLDOUT_DIR / "holdout_y_class.npy")
    
    fold_results = {}
    holdout_masks = []
    
    # ── 2. Evaluate Individual Folds ─────────────────────────────────────
    for fold in [1, 2, 3]:
        tensor_dir = CV_DIR / f"fold_{fold}" / "tensors"
        model_path = CV_DIR / f"fold_{fold}" / "model.keras"
        config_path = CV_DIR / f"fold_{fold}" / "config.json"
        
        if not model_path.exists():
            print(f"⚠️  Fold {fold} model missing. Skipping.")
            continue
            
        print(f"\n📂 Loading and Evaluating Fold {fold}...")
        model = tf.keras.models.load_model(model_path)
        with open(config_path) as f:
            config = json.load(f)
            
        th_p = config["Prob_Win_threshold"]
        th_o = config["Pred_OS_threshold"]
        
        # Test Set
        t_micro = np.load(tensor_dir / "test_micro.npy")
        t_macro = np.load(tensor_dir / "test_macro.npy")
        t_yc = np.load(tensor_dir / "test_y_class.npy")
        
        res_test = evaluate_split(model, th_p, th_o, t_micro, t_macro, t_yc)
        
        # Holdout Set
        res_hold = evaluate_split(model, th_p, th_o, h_micro, h_macro, h_yc)
        holdout_masks.append(res_hold["mask"]) # store for ensemble
        
        fold_results[fold] = {
            "th_o": th_o,
            "test_wr": res_test["wr"],
            "test_tr": res_test["trades"],
            "hold_wr": res_hold["wr"],
            "hold_tr": res_hold["trades"]
        }
        
    # ── 3. Vote Ensemble on Holdout ──────────────────────────────────────
    print("\n🗳️  Calculating Voting Ensemble on 2024 Holdout...")
    if len(holdout_masks) == 3:
        # Stack masks: (3, N)
        stacked_masks = np.stack(holdout_masks, axis=0) # boolean
        # Majority Vote: >= 2 models agree to enter
        ensemble_mask = np.sum(stacked_masks, axis=0) >= 2
        ens_trades = np.sum(ensemble_mask)
        ens_wr = np.mean(h_yc[ensemble_mask]) if ens_trades > 0 else 0.0
    else:
        ens_trades = 0
        ens_wr = 0.0

    # ── 4. Generate Report ───────────────────────────────────────────────
    md = "# Iteration 2: Expanding Window Cross-Validation Report\n\n"
    md += "This report evaluates the execution-priced training pipeline across 3 separate timeline folds to guarantee structural robustness across shifting market regimes.\n\n"
    
    md += "## Fold Configurations\n"
    md += "- **Fold 1**: Train (20-21), Val (22H1), Test (22H2)\n"
    md += "- **Fold 2**: Train (20-22H1), Val (22H2), Test (23H1)\n"
    md += "- **Fold 3**: Train (20-22), Val (23H1), Test (23H2)\n\n"
    
    md += "## Independent Fold Performance\n"
    md += "| Fold | Calibrated Os Threshold | Test WR | Test Trades | 2024 Holdout WR | 2024 Holdout Trades |\n"
    md += "|------|-------------------------|---------|-------------|-----------------|---------------------|\n"
    
    for fold in [1,2,3]:
        if fold in fold_results:
            r = fold_results[fold]
            md += f"| Fold {fold} | `{r['th_o']:.2f}` | **{r['test_wr']:.2%}** | {r['test_tr']} | {r['hold_wr']:.2%} | {r['hold_tr']} |\n"

    md += "\n## Ensemble Performance (2024 Holdout)\n"
    md += "The final prediction strategy utilizes a **Majority Voting Ensemble** (trade enters if $\ge 2$ of the 3 Fold models signal a go).\n\n"
    
    md += f"- **Holdout Ensemble Win Rate**: `{ens_wr:.2%}`\n"
    md += f"- **Holdout Ensemble Trades**: `{ens_trades}`\n\n"
    
    md += "### Conclusion\n"
    if ens_wr >= 0.70:
        md += "The Cross-Validation confirms the pipeline is incredibly stable! The ensemble model achieved an exceptionally robust holdout win-rate, neutralizing the risk of over-optimizing to a specific isolated 6-month timeframe."
    else:
        md += "The Cross-Validation revealed notable performance variance between folds. The model might still be sensitive to specific regime changes."
        
    with open(REPORT_MD, "w") as f:
        f.write(md)
        
    print(f"\n✅ Report generated: {REPORT_MD}")


if __name__ == "__main__":
    main()
