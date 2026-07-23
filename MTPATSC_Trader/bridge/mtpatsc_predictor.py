"""
MTPATSC Trader — Setup Classifier Predictor
=============================================
Loads the MTPATSC inference model (single Keras model) and applies
calibrated probability thresholds to generate trade signals.

Replaces the old 3-fold ensemble predictor from BrickOfTicks_Trader.

Model input:  4 tensors [ancs_fine, ancs_coarse, history, scalars]
Model output: 5-class softmax [P(T0), P(T1), P(T2), P(T3), P(T4)]

Signal logic:
  1. Veto if P(T0) > veto_threshold
  2. Find argmax class
  3. If predicted class threshold >= 1.0 → setup disabled, skip
  4. If P(predicted_class) < threshold → below confidence, skip
  5. Otherwise → SIGNAL with setup_type and direction logic
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Suppress excessive TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
except ImportError:
    tf = None
    logger.critical("TensorFlow is not installed. Model cannot be loaded.")

import numpy as np


class MTPatscPredictor:
    """
    Single-model 5-class setup classifier predictor.

    Loads model_inference.keras and config.json from the models directory.
    On each brick close, takes the 4 feature tensors and returns a trade signal.
    """

    def __init__(self, models_dir: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = Path(models_dir) if models_dir else Path(base_dir) / "models"
        self.model = None
        self.config = {}
        self.thresholds = {}
        self.veto_threshold = 0.40
        self.rr_profiles = {1: 1.0, 2: 2.0, 3: 2.0, 4: 3.0}

    def load(self):
        """
        Load the Keras inference model and calibration config.
        Exits with code 1 if model fails to load.
        """
        if tf is None:
            logger.critical("Cannot load model: TensorFlow is not installed.")
            sys.exit(1)

        model_path = self.models_dir / "model_inference.keras"
        config_path = self.models_dir / "config.json"

        if not model_path.exists():
            logger.critical(f"Model not found at {model_path}")
            sys.exit(1)

        if not config_path.exists():
            logger.critical(f"Config not found at {config_path}")
            sys.exit(1)

        # Load model
        try:
            logger.info(f"Loading MTPATSC inference model from {model_path}...")
            self.model = tf.keras.models.load_model(str(model_path), compile=False)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load model: {e}")
            sys.exit(1)

        # Load thresholds
        with open(config_path) as f:
            self.config = json.load(f)

        self.thresholds = {
            1: self.config.get("T1_threshold", 1.0),
            2: self.config.get("T2_threshold", 1.0),
            3: self.config.get("T3_threshold", 1.0),
            4: self.config.get("T4_threshold", 1.0),
        }
        self.veto_threshold = self.config.get("T0_veto_threshold", 0.40)

        logger.info(f"Thresholds: T1={self.thresholds[1]:.2f}, T2={self.thresholds[2]:.2f}, "
                     f"T3={self.thresholds[3]:.2f}, T4={self.thresholds[4]:.2f}, "
                     f"Veto={self.veto_threshold:.2f}")

        # Log which setups are active
        for sc in [1, 2, 3, 4]:
            status = "ACTIVE" if self.thresholds[sc] < 1.0 else "DISABLED"
            logger.info(f"  T{sc}: {status} (threshold={self.thresholds[sc]:.2f}, RR=1:{self.rr_profiles[sc]:.0f})")

    def predict(self, tensors: Dict[str, np.ndarray], brick_direction: int) -> Dict[str, Any]:
        """
        Run inference and apply threshold/veto logic.

        Args:
            tensors: Dict with keys 'ancs_fine', 'ancs_coarse', 'history', 'scalars'
                     Each is a numpy array with batch dim = 1.
            brick_direction: 1 (UP) or -1 (DOWN) from the brick event.

        Returns:
            dict with:
                'action':   0 (skip) or 1 (trade)
                'setup_type': int (1-4) or 0 if no trade
                'probs':    np.ndarray of shape (5,) — raw probabilities
                'direction': int — actual trade direction (may differ for T3/T4)
                'rr':       float — risk:reward ratio
                'reason':   str — why skipped (if action=0)
        """
        if self.model is None:
            logger.warning("Predict called but model not loaded. Returning SKIP.")
            return {'action': 0, 'setup_type': 0, 'probs': np.zeros(5),
                    'direction': 0, 'rr': 0, 'reason': 'model_not_loaded'}

        # Run inference
        try:
            probs = self.model(
                [tensors['ancs_fine'], tensors['ancs_coarse'],
                 tensors['history'], tensors['scalars']],
                training=False
            ).numpy().flatten()
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {'action': 0, 'setup_type': 0, 'probs': np.zeros(5),
                    'direction': 0, 'rr': 0, 'reason': f'inference_error: {e}'}

        # Log probabilities
        logger.debug(f"Probs: T0={probs[0]:.3f} T1={probs[1]:.3f} T2={probs[2]:.3f} "
                      f"T3={probs[3]:.3f} T4={probs[4]:.3f}")

        # 1. Veto check
        if probs[0] > self.veto_threshold:
            return {'action': 0, 'setup_type': 0, 'probs': probs,
                    'direction': 0, 'rr': 0, 'reason': f'vetoed (P(T0)={probs[0]:.3f})'}

        # 2. Find predicted class
        pred_class = int(np.argmax(probs))
        if pred_class == 0:
            return {'action': 0, 'setup_type': 0, 'probs': probs,
                    'direction': 0, 'rr': 0, 'reason': 'predicted T0'}

        # 3. Check if setup is enabled
        if pred_class not in self.thresholds:
            return {'action': 0, 'setup_type': 0, 'probs': probs,
                    'direction': 0, 'rr': 0, 'reason': f'unknown class {pred_class}'}

        threshold = self.thresholds[pred_class]
        if threshold >= 1.0:
            return {'action': 0, 'setup_type': pred_class, 'probs': probs,
                    'direction': 0, 'rr': 0,
                    'reason': f'T{pred_class} disabled (threshold=1.0)'}

        # 4. Confidence check
        if probs[pred_class] < threshold:
            return {'action': 0, 'setup_type': pred_class, 'probs': probs,
                    'direction': 0, 'rr': 0,
                    'reason': f'below threshold (P(T{pred_class})={probs[pred_class]:.3f} < {threshold:.3f})'}

        # 5. SIGNAL! Determine trade direction
        trade_direction = brick_direction
        if pred_class in (3, 4):
            # T3/T4 are reversals — trade against the brick direction
            trade_direction = -brick_direction

        rr = self.rr_profiles[pred_class]

        return {
            'action': 1,
            'setup_type': pred_class,
            'probs': probs,
            'direction': trade_direction,
            'rr': rr,
            'reason': f'SIGNAL T{pred_class} (P={probs[pred_class]:.3f})'
        }
