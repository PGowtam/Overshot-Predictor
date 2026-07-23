import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Evaluator")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "setup_classifier"
TENSOR_DIR = OUTPUT_DIR / "tensors"

def main():
    test_path = TENSOR_DIR / "test.npz"
    model_path = OUTPUT_DIR / "model_inference.keras"
    config_path = OUTPUT_DIR / "config.json"
    
    if not test_path.exists() or not model_path.exists() or not config_path.exists():
        logger.error("Required test tensors, model, or calibration config not found.")
        return
        
    logger.info("Loading test data, inference model, and calibration config...")
    test_data = np.load(test_path)
    model = tf.keras.models.load_model(str(model_path))
    
    with open(config_path) as f:
        config = json.load(f)
        
    # 1. Generate predictions
    x_test = [
        test_data['ancs_fine'],
        test_data['ancs_coarse'],
        test_data['history'],
        test_data['scalars']
    ]
    
    probs = model.predict(x_test) # Shape: (N, 5)
    pred_classes = np.argmax(probs, axis=1)
    test_labels = test_data['label']
    
    # Extract binary win flags
    win_flags = {
        1: test_data['t1_win'],
        2: test_data['t2_win'],
        3: test_data['t3_win'],
        4: test_data['t4_win']
    }
    rr_profiles = {1: 1.0, 2: 2.0, 3: 2.0, 4: 3.0}
    
    # 2. Compute General Classification Metrics
    # Top-2 Accuracy
    top2_correct = 0
    for i in range(len(test_labels)):
        top2_idx = np.argsort(probs[i])[-2:]
        if test_labels[i] in top2_idx:
            top2_correct += 1
    top2_acc = float(top2_correct / len(test_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, pred_classes).tolist()
    
    # 3. Evaluate Calibrated Trading Performance
    veto_threshold = config.get("T0_veto_threshold", 0.40)
    per_class_performance = {}
    total_oos_return = 0.0
    total_trades_count = 0
    
    # Array to track individual trade returns on the test set
    trade_returns = np.zeros(len(test_labels))
    
    logger.info("Evaluating trading performance on OOS Test Split...")
    for setup_class in [1, 2, 3, 4]:
        theta = config.get(f"T{setup_class}_threshold", 1.0)
        rr = rr_profiles[setup_class]
        t_win = win_flags[setup_class]
        
        if theta >= 1.0:
            per_class_performance[f"T{setup_class}"] = {
                "threshold": theta,
                "n_trades": 0,
                "win_rate": 0.0,
                "ev": 0.0,
                "total_return_R": 0.0
            }
            logger.info(f"T{setup_class} is disabled (threshold = 1.0)")
            continue
            
        mask = (pred_classes == setup_class) & (probs[:, setup_class] >= theta) & (probs[:, 0] <= veto_threshold)
        n_trades = int(mask.sum())
        
        if n_trades > 0:
            win_rate = float(np.mean(t_win[mask]))
            ev = win_rate * rr - (1.0 - win_rate) * 1.0
            
            # PnL for each trade: +rr for win, -1 for loss
            pnl_vector = np.where(t_win == 1.0, rr, -1.0)
            trade_returns[mask] = pnl_vector[mask]
            
            total_return_R = float(trade_returns[mask].sum())
            total_oos_return += total_return_R
            total_trades_count += n_trades
            
            per_class_performance[f"T{setup_class}"] = {
                "threshold": theta,
                "n_trades": n_trades,
                "win_rate": round(win_rate, 4),
                "ev": round(ev, 4),
                "total_return_R": round(total_return_R, 2)
            }
            logger.info(f"T{setup_class} Calibrated (theta={theta:.2f}): Trades={n_trades}, WR={win_rate:.2%}, EV={ev:.2f} R, Return={total_return_R:.2f} R")
        else:
            per_class_performance[f"T{setup_class}"] = {
                "threshold": theta,
                "n_trades": 0,
                "win_rate": 0.0,
                "ev": 0.0,
                "total_return_R": 0.0
            }
            logger.info(f"T{setup_class} Calibrated (theta={theta:.2f}): No trades triggered")
            
    combined_ev = float(total_oos_return / len(test_labels))
    logger.info(f"Combined total return: {total_oos_return:.2f} R across {total_trades_count} trades (EV: {combined_ev:.4f} R per observed brick)")
    
    # 4. Monthly Regime Breakdown
    date_ints = test_data['date_ints']
    unique_months = np.unique(date_ints)
    monthly_performance = {}
    
    for month in sorted(unique_months):
        month_mask = date_ints == month
        month_trades = (trade_returns != 0.0) & month_mask
        
        n_month_trades = int(month_mask[trade_returns != 0.0].sum())
        month_return = float(trade_returns[month_trades].sum())
        
        monthly_performance[str(month)] = {
            "n_trades": n_month_trades,
            "total_return_R": round(month_return, 2),
            "bricks_count": int(month_mask.sum())
        }
        logger.info(f"Month {month}: Trades={n_month_trades}, Return={month_return:.2f} R (Bricks observed: {month_mask.sum()})")
        
    # 5. Limit Fill Sensitivity Analysis (Monte Carlo Simulation - 100 runs)
    fill_sensitivity = {}
    t2_indices = np.where((pred_classes == 2) & (probs[:, 2] >= config.get("T2_threshold", 1.0)) & (probs[:, 0] <= veto_threshold))[0]
    t4_indices = np.where((pred_classes == 4) & (probs[:, 4] >= config.get("T4_threshold", 1.0)) & (probs[:, 0] <= veto_threshold))[0]
    
    logger.info("Running Limit Fill Sensitivity Monte Carlo simulation...")
    for fill_rate in [1.0, 0.95, 0.90, 0.80, 0.70]:
        sim_returns = []
        for _ in range(100):
            temp_returns = trade_returns.copy()
            
            # Apply fill rejections for T2
            if len(t2_indices) > 0:
                t2_fills = np.random.random(len(t2_indices)) < fill_rate
                unfilled_t2 = t2_indices[~t2_fills]
                temp_returns[unfilled_t2] = 0.0
                
            # Apply fill rejections for T4
            if len(t4_indices) > 0:
                t4_fills = np.random.random(len(t4_indices)) < fill_rate
                unfilled_t4 = t4_indices[~t4_fills]
                temp_returns[unfilled_t4] = 0.0
                
            sim_returns.append(temp_returns.sum())
            
        avg_pnl = float(np.mean(sim_returns))
        avg_ev = float(avg_pnl / len(test_labels))
        fill_sensitivity[f"fill_rate_{int(fill_rate*100)}"] = {
            "avg_total_return_R": round(avg_pnl, 2),
            "avg_ev": round(avg_ev, 4)
        }
        logger.info(f"Fill Rate {int(fill_rate*100)}%: Avg Return = {avg_pnl:.2f} R, Avg EV = {avg_ev:.4f} R")
        
    # 6. Save Evaluation Report
    report = {
        "top2_accuracy": round(top2_acc, 4),
        "confusion_matrix": cm,
        "combined_total_return_R": round(total_oos_return, 2),
        "combined_ev_per_brick": round(combined_ev, 4),
        "total_trades_taken": total_trades_count,
        "per_class_performance": per_class_performance,
        "monthly_performance": monthly_performance,
        "fill_sensitivity": fill_sensitivity
    }
    
    report_path = OUTPUT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved complete evaluation report to {report_path}")

if __name__ == "__main__":
    main()
