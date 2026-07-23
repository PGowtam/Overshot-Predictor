import numpy as np
from collections import deque
from typing import Optional, Tuple, List

class InferenceBuffer:
    """
    Stateful buffer that collects 100-tick micro sequences and 10-brick macro sequences
    to format tensors for the Keras ensemble models.
    """
    def __init__(self):
        # Micro queue tracks up to 100 ticks. Format: (feature_vector: List[float], brick_id: int)
        self.micro = deque(maxlen=100)
        
        # Snapshots: Stores up to 10 brick-closing micro snapshots, shape (100, 9)
        self.snapshots = deque(maxlen=10)
        
        # Macro: Stores up to 10 macro vectors, shape (3)
        self.macro = deque(maxlen=10)

    def append_tick(self, feat_vec: List[float], current_brick_id: int):
        """
        Push a new tick's feature vector into the rolling micro buffer.
        """
        self.micro.append((feat_vec, current_brick_id))

    def on_brick_close(self, current_brick_id: int, macro_vec: List[float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Called when a brick is completed. Generates a snapshot of the micro buffer,
        stores the macro vector, and if history is full, returns the Tensor inputs.
        
        Returns:
            Tuple of (micro_tensor, macro_tensor) or None if < 10 bricks in history.
            micro_tensor shape: (1, 10, 100, 9)
            macro_tensor shape: (1, 10, 3)
        """
        if len(self.micro) == 0:
            return None

        # Create a 100x9 zero-padded array for the snapshot
        snapshot = np.zeros((100, 9), dtype=np.float32)
        
        # Write the ticks to the end of the snapshot (zero-padding at front)
        start_idx = 100 - len(self.micro)
        
        for i, (vec, b_id) in enumerate(self.micro):
            vec_copy = list(vec)
            
            # Rewrite Flag_Curr (idx 6): 1.0 if tick belongs to the just-closed brick
            vec_copy[6] = 1.0 if b_id == current_brick_id else 0.0
            
            # Rewrite Decay (idx 8): Age of the tick measured in bricks
            vec_copy[8] = float(current_brick_id - b_id) / 100.0
            
            snapshot[start_idx + i] = vec_copy

        self.snapshots.append(snapshot)
        self.macro.append(macro_vec)

        # Gate check: wait for 10 full bricks of history
        if len(self.snapshots) == 10:
            micro_tensor = np.array(self.snapshots, dtype=np.float32)[np.newaxis, ...]  # (1, 10, 100, 9)
            macro_tensor = np.array(self.macro, dtype=np.float32)[np.newaxis, ...]      # (1, 10, 3)
            return micro_tensor, macro_tensor
        
        return None
