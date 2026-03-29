
from stable_baselines3 import DQN
import numpy as np
from config.definitions import M_DQN_PATH
from utils.logger import logger
import os

class EnsembleAgent:
    def __init__(self):
        self.model = None
        
    def load_all(self):
        logger.info("Loading Single DQN Agent (XAUUSD)...")
        path = M_DQN_PATH
        
        # Determine full path with zip extension check
        if os.path.exists(path + ".zip"):
            full_path = path + ".zip"
        elif os.path.exists(path):
            full_path = path
        else:
            logger.error(f"DQN Model not found at {path}")
            return

        try:
            self.model = DQN.load(full_path)
            logger.info("Loaded DQN Model successfully.")
        except Exception as e:
            logger.error(f"Failed to load DQN Model: {e}")

    def predict(self, obs, lstm_states=None, episode_starts=None, obs_stack=None):
        """
        Returns: 
            final_action (int): 1 (Buy/Trade) or 0 (Skip)
            new_lstm_states: None (DQN is stateless)
            score: Confidence (fake score based on action)
        """
        if self.model is None:
            logger.warning("Predict called but DQN model not loaded. Returning 0.")
            return 0, None, 0.0
            
        # DQN Predict
        action, _ = self.model.predict(obs, deterministic=True)
        final_action = int(action)
        
        # Score for logging (1.0 for Buy, -1.0 for Hold/Sell)
        score = 1.0 if final_action == 1 else -1.0
        
        return final_action, None, score
