
import numpy as np
import torch
import os
from Agents.iql_agent import IQLAgent

class MetaController:
    def __init__(self, device="cpu"):
        self.device = device
        
        # Paths
        self.TREND_MODEL = "Models/trend_agent.pt"
        self.CHOP_MODEL = "Models/chop_agent.pt"
        self.HIGHVOL_MODEL = "Models/high_vol_agent.pt"
        self.TRANSITION_MODEL = "Models/transition_agent.pt"
        self.META_MODEL = "Models/meta_agent.pt"
        
        # Dimensions (Hardcoded based on training)
        self.obs_dim = 30  # Raw feature dim
        self.meta_obs_dim = 4 + 4 + 4 # 4 actions + 4 confs + 4 regime probs = 12
        self.hidden_dim = 128
        
        # Initialize Agents
        self.trend = self._load_agent(self.TREND_MODEL, self.obs_dim)
        self.chop = self._load_agent(self.CHOP_MODEL, self.obs_dim)
        self.highvol = self._load_agent(self.HIGHVOL_MODEL, self.obs_dim)
        self.transition = self._load_agent(self.TRANSITION_MODEL, self.obs_dim)
        
        self.meta = self._load_agent(self.META_MODEL, self.meta_obs_dim)
        
        self.specialists = [self.trend, self.chop, self.highvol, self.transition]
        self.regime_names = ["Trend", "Chop", "HighVol", "Transition"]

    def _load_agent(self, path, input_dim):
        agent = IQLAgent(input_dim, hidden_dim=self.hidden_dim, device=self.device)
        if os.path.exists(path):
            agent.load_state_dict(torch.load(path, map_location=self.device))
            agent.q.eval()
        else:
            raise FileNotFoundError(f"Model not found: {path}")
        return agent

    def act(self, obs):
        """
        Full forward pass:
        1. Specialists act on raw obs.
        2. Construct meta_obs.
        3. Meta-Agent acts on meta_obs.
        """
        # 1. Specialists
        actions = []
        confidences = []
        
        for agent in self.specialists:
            a, c = agent.act(obs)
            actions.append(a)
            confidences.append(c)
            
        # 2. Construct Meta Obs
        # Structure: [Actions (4), Confidences (4), Regime Probs (4)]
        # Regime probs are the last 4 elements of raw obs
        regime_probs = obs[-4:]
        
        meta_obs_input = np.array(
            actions + confidences + list(regime_probs),
            dtype=np.float32
        )
        
        # 3. Meta Decision
        meta_action, meta_confidence = self.meta.act(meta_obs_input)
        
        # Metadata for logging
        regime_idx = np.argmax(regime_probs)
        regime = self.regime_names[regime_idx]
        disagreement = np.std(actions)
        
        info = {
            "meta_confidence": meta_confidence,
            "specialist_actions": actions,
            "specialist_confidences": confidences,
            "regime": regime,
            "disagreement": disagreement,
            "trend_action": actions[0] # Useful for comparison
        }
        
        return meta_action, info
