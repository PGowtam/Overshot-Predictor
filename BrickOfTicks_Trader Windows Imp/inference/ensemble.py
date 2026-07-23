"""
Phase 5: Ensemble Inference

Loads the 3 fold models, runs inference on the tensor inputs, applies majority voting,
and enforces "Baiting" logic for high-confidence reversal trades.
"""
import json
import tensorflow as tf
from pathlib import Path

from BrickOfTicks_Trader.config.settings import (
    PROB_WIN_THRESHOLD, PRED_OS_THRESHOLD,
    BAIT_PROB_WIN_THRESHOLD, BAIT_PRED_OS_THRESHOLD,
    ENSEMBLE_VOTE_THRESHOLD
)
from BrickOfTicks_Trader.utils.logger import logger


class EnsemblePredictor:
    def __init__(self, models_dir: str):
        self.models = []
        self.configs = []
        self.models_dir = Path(models_dir)

    def load(self):
        """Load 3 fold models and their configs."""
        logger.info(f"Loading ensemble models from {self.models_dir}")
        for fold in [1, 2, 3]:
            fold_dir = self.models_dir / f"fold_{fold}"
            model_path = fold_dir / "model.keras"
            config_path = fold_dir / "config.json"

            if not model_path.exists() or not config_path.exists():
                logger.error(f"Missing model or config for fold {fold} at {fold_dir}")
                raise FileNotFoundError(f"Missing model/config in {fold_dir}")

            model = tf.keras.models.load_model(model_path)
            with open(config_path, "r") as f:
                config = json.load(f)

            self.models.append(model)
            self.configs.append(config)
            logger.info(f"Loaded Fold {fold}: Thresholds config={config}")

    def predict(self, micro_tensor, macro_tensor) -> dict:
        """
        Run all 3 models and apply majority voting.
        
        Args:
            micro_tensor: (1, 10, 100, 9)
            macro_tensor: (1, 10, 3)
            
        Returns:
            {
                "action": 1 (ENTER) or -1 (REVERSE) or 0 (SKIP),
                "votes": int,
                "details": list of dicts with model specific predictions,
                "trade_type": "standard", "baiting", or "none"
            }
        """
        if not self.models:
            raise RuntimeError("Models not loaded. Call load() first.")

        votes = 0
        details = []

        for i, (model, config) in enumerate(zip(self.models, self.configs)):
            # training=False to disable dropout/batch_norm behavior
            preds = model([micro_tensor, macro_tensor], training=False)
            
            # Extract Head A (prob_win) and Head B (pred_os)
            prob_win = float(preds[0].numpy().flatten()[0])
            pred_os = float(preds[1].numpy().flatten()[0])

            # The config holds per-fold thresholds, but settings.py has unified ones we override with, 
            # based on updated requirements if they exist. Here we use the settings.py thresholds 
            # if they are defined, else fallback to config. We use config values as per spec.
            th_p = config.get("Prob_Win_threshold", PROB_WIN_THRESHOLD)
            th_o = config.get("Pred_OS_threshold", PRED_OS_THRESHOLD)

            signal = (prob_win >= th_p) and (pred_os >= th_o)
            if signal:
                votes += 1

            details.append({
                "fold": i + 1,
                "prob_win": prob_win,
                "pred_os": pred_os,
                "signal": signal
            })

        # ── Baiting Logic ─────────────────────────────────
        # Criteria: ALL folds must have prob_win < BAIT_PROB_WIN_THRESHOLD and pred_os < BAIT_PRED_OS_THRESHOLD
        is_bait = all(
            d["prob_win"] < BAIT_PROB_WIN_THRESHOLD and d["pred_os"] < BAIT_PRED_OS_THRESHOLD 
            for d in details
        )

        if votes >= ENSEMBLE_VOTE_THRESHOLD:
            action = 1
            trade_type = "standard"
        elif is_bait:
            action = -1
            trade_type = "baiting"
        else:
            action = 0
            trade_type = "none"

        return {
            "action": action,
            "votes": votes,
            "details": details,
            "trade_type": trade_type
        }
