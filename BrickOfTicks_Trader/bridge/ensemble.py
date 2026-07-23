import os
import sys
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Supress excessive TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
except ImportError:
    tf = None
    logger.critical("TensorFlow is not installed. Models cannot be loaded.")

class EnsemblePredictor:
    """
    Manages a 3-fold ensemble of deep learning models.
    Takes micro and macro tensors, generates predictions, and uses majority voting
    to emit a trading signal without any baiting logic.
    """
    PROB_WIN_THRESHOLD = 0.5
    PRED_OS_THRESHOLD  = 1.4   # Calibrated on K=0.00295 holdout
    VOTE_THRESHOLD     = 2     # >= 2 out of 3

    def __init__(self, primary_dir=None, fallback_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.primary_dir = primary_dir if primary_dir else os.path.join(base_dir, "models")
        self.fallback_dir = fallback_dir if fallback_dir else os.path.join(base_dir, "..", "outputs", "exec", "cv")
        self.models = []

    def load(self):
        """
        Load the 3 Keras models from disk. Exits with code 1 if any model fails.
        """
        if tf is None:
            logger.critical("Cannot load models: TensorFlow is not installed.")
            sys.exit(1)
            
        for i in range(1, 4):
            # Attempt primary path first
            path = os.path.join(self.primary_dir, f"fold_{i}", "model.keras")
            
            if not os.path.exists(path):
                # Attempt fallback
                path = os.path.join(self.fallback_dir, f"fold_{i}", "model.keras")
            
            if not os.path.exists(path):
                logger.critical(f"Model file not found for fold {i} at {path}")
                sys.exit(1)
                
            try:
                logger.info(f"Loading Keras model for fold {i} from {path}...")
                model = tf.keras.models.load_model(path, compile=False)
                self.models.append(model)
            except Exception as e:
                logger.critical(f"Failed to load model {path}: {e}")
                sys.exit(1)

        logger.info(f"Successfully loaded {len(self.models)} models.")

    def predict(self, micro_tensor: Any, macro_tensor: Any) -> Dict[str, Any]:
        """
        Runs inference across all 3 models and applies threshold voting.
        
        Returns:
            dict with 'action' (1 or 0), 'votes' (0 to 3), and 'details'
        """
        if not self.models:
            logger.warning("Predict called but models not loaded. Returning SKIP.")
            return {'action': 0, 'votes': 0, 'details': []}

        votes = 0
        details = []

        for model in self.models:
            # Model output: [prob_win, pred_os]
            preds = model([micro_tensor, macro_tensor], training=False)
            
            pw = float(preds[0].numpy().flatten()[0])
            po = float(preds[1].numpy().flatten()[0])
            
            signal = (pw >= self.PROB_WIN_THRESHOLD) and (po >= self.PRED_OS_THRESHOLD)
            votes += int(signal)
            
            details.append({'prob_win': pw, 'pred_os': po, 'signal': signal})

        # Standard signal only — NO BAITING (action=-1 path completely removed)
        action = 1 if votes >= self.VOTE_THRESHOLD else 0
        
        return {'action': action, 'votes': votes, 'details': details}
