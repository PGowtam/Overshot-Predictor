
import numpy as np
import joblib
import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger
from models.meta_controller import MetaController # Assumes we copied it or will create it here

class MetaTraderWrapper:
    def __init__(self):
        self.controller = None
        
    def load_all(self):
        logger.info("Initializing Meta-Controller System...")
        try:
            # We assume MetaController is in the same folder or path
            # We need to make sure MetaController can find its models relative to Main
            # We might need to adjust paths inside MetaController or set CWD in main
            self.controller = MetaController()
            logger.info("Meta-Controller Loaded Successfully.")
        except Exception as e:
            logger.error(f"Failed to load Meta-Controller: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def predict(self, obs, lstm_states=None, episode_starts=None, obs_stack=None):
        """
        Adapts MetaController output to OrbitEngine interface.
        
        Args:
            obs (np.array): 30-dim vector (Required!)
            lstm_states: Ignored by IQL
            episode_starts: Ignored
            obs_stack: Ignored (MetaController handles its own context if needed, but it's largely reactive)
            
        Returns:
            final_action (int): 0 or 1
            new_lstm_states: None
            score: Confidence or Meta-Action value
        """
        if self.controller is None:
            logger.warning("Predict called but MetaController not loaded.")
            return 0, None, 0.0
            
        try:
            # MetaController.act returns (action, info_dict)
            action, info = self.controller.act(obs)
            
            # Extract confidence for logging
            score = info.get("meta_confidence", 0.0)
            
            # Log detailed info
            if action == 1:
                logger.info(f"META TAKE | Conf: {score:.2f} | Regime: {info.get('regime')} | Disagreement: {info.get('disagreement'):.4f}")
            
            return action, None, score
            
        except Exception as e:
            logger.error(f"Meta-Prediction Error: {e}")
            return 0, None, 0.0
