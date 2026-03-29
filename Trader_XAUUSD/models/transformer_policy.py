
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor that applies a Transformer Encoder to stacked observations.
    
    Expected Input: (Batch, N_Stack * Obs_Dim) (Flattened Stack)
    Process: 
        1. Reshape -> (Batch, N_Stack, Obs_Dim)
        2. Linear Projection -> (Batch, N_Stack, D_Model)
        3. Positional Encoding
        4. Transformer Encoder
        5. Flatten -> (Batch, N_Stack * D_Model) or Global Average Pooling
    """
    
    def __init__(self, observation_space: gym.spaces.Box, n_stack: int = 10, d_model: int = 64, n_head: int = 4, n_layers: int = 2):
        # We need to know the original feature dim (before stacking)
        # obs_dim = total_dim / n_stack
        total_dim = observation_space.shape[0]
        original_dim = total_dim // n_stack
        
        # Calculate output dim
        # If we flatten: d_model * n_stack
        # If we pool: d_model
        output_dim = d_model * n_stack
        
        super(TransformerExtractor, self).__init__(observation_space, features_dim=output_dim)
        
        self.n_stack = n_stack
        self.original_dim = original_dim
        self.d_model = d_model
        
        # 1. Input Projection
        self.input_net = nn.Sequential(
            nn.Linear(original_dim, d_model),
            nn.ReLU()
        )
        
        # 2. Positional Encoding (Learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_stack, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model*4, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Output Projection (Optional, usually Flatten is enough)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input Shape: (Batch, N_Stack * Obs_Dim)
        batch_size = observations.shape[0]
        
        # 1. Reshape to Sequence
        # (Batch, N_Stack, Obs_Dim)
        x = observations.view(batch_size, self.n_stack, self.original_dim)
        
        # 2. Project to Embedding Dimension
        x = self.input_net(x) # (Batch, N_Stack, D_Model)
        
        # 3. Add Positional Encoding
        x = x + self.pos_embedding
        
        # 4. Transformer Pass
        x = self.transformer(x) # (Batch, N_Stack, D_Model)
        
        # 5. Flatten
        x = x.flatten(start_dim=1) # (Batch, N_Stack * D_Model)
        
        return x
