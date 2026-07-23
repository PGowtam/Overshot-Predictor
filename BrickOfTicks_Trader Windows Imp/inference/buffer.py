"""
Phase 4: Micro-Buffer & Tensor Assembly

Maintains the continuous tick buffer and assembles model inputs on brick close.

CRITICAL INVARIANTS:
- micro_buffer NEVER resets at brick boundaries (continuous stream)
- Stores (9D_vector, brick_id) tuples
- On brick close: snapshot → rewrite Flag_Curr & Decay → zero-pad → macro vector
- Returns None until 10 bricks in history

Tensor shapes:
- micro: (1, 10, 100, 9)
- macro: (1, 10, 3)
"""

import numpy as np
from collections import deque
from math import log

BUFFER_SIZE = 100     # micro_buffer maxlen (ticks per snapshot)
MACRO_HISTORY = 10    # number of brick snapshots for model input


class InferenceBuffer:
    """
    Continuous micro-buffer + macro-history for tensor assembly.

    Usage:
        1. For every tick: call append_tick(feature_vector_9d, brick_id)
        2. On every brick close: call on_brick_close(brick)
           - Returns (micro_tensor, macro_tensor) or None if < 10 bricks
    """

    def __init__(self):
        # Micro-buffer: stores (9D_vector, brick_id) tuples
        # NEVER reset at brick boundaries
        self.micro_buffer = deque(maxlen=BUFFER_SIZE)
        self.current_brick_id = 0

        # Macro-history: stores 3D macro vectors per brick
        self.macro_history = deque(maxlen=MACRO_HISTORY)

        # Brick size history for z_size calculation
        self.brick_size_history = []

        # Snapshot storage: last 10 (snapshot_array, timestamp) pairs
        self.snapshots = deque(maxlen=MACRO_HISTORY)

    def append_tick(self, feature_vector_9d: list, brick_id: int):
        """Append a tick's 9D feature vector to the continuous buffer."""
        self.micro_buffer.append(
            (np.array(feature_vector_9d, dtype=np.float32), brick_id)
        )

    def on_brick_close(self, brick) -> tuple:
        """
        Called when a new brick closes.

        Steps:
          1. Increment brick ID
          2. Compute macro vector for this brick
          3. Snapshot the micro buffer (copy + rewrite Flag_Curr/Decay)
          4. Zero-pad if < 100 ticks
          5. Stack last 10 snapshots → (1, 10, 100, 9)
          6. Stack last 10 macro vectors → (1, 10, 3)

        Returns:
            (micro_tensor, macro_tensor) or None if < 10 bricks
        """
        self.current_brick_id += 1

        # ── 1. Macro Vector ──────────────────────────────
        # Duration: time since previous brick close
        if self.snapshots:
            prev_ts = self.snapshots[-1][1]
        else:
            prev_ts = brick.timestamp
        duration_s = max(0, (brick.timestamp - prev_ts) / 1000.0)
        log_dur = log(duration_s + 1)

        # Direction
        direction = 1.0 if brick.uptrend else -1.0

        # z_size: (brick_size - mean_50) / std_50
        self.brick_size_history.append(brick.brick_size)
        recent_sizes = self.brick_size_history[-50:]
        if len(recent_sizes) < 2:
            z_size = 0.0
        else:
            mu = np.mean(recent_sizes)
            sigma = np.std(recent_sizes, ddof=1)
            z_size = (brick.brick_size - mu) / sigma if sigma > 1e-12 else 0.0

        macro_vec = np.array([log_dur, direction, z_size], dtype=np.float32)
        self.macro_history.append(macro_vec)

        # ── 2. Micro Snapshot ────────────────────────────
        buf_len = len(self.micro_buffer)
        if buf_len == 0:
            snapshot = np.zeros((BUFFER_SIZE, 9), dtype=np.float32)
        else:
            vectors = []
            brick_ids = []
            for vec, bid in self.micro_buffer:
                vectors.append(vec.copy())
                brick_ids.append(bid)

            arr = np.stack(vectors)  # (buf_len, 9)

            # Rewrite Flag_Curr (index 6) and Decay (index 8)
            for i in range(len(arr)):
                arr[i, 6] = 1.0 if brick_ids[i] == self.current_brick_id else 0.0
                arr[i, 8] = min(
                    (self.current_brick_id - brick_ids[i]) / BUFFER_SIZE, 1.0
                )

            # Zero-pad at front if < BUFFER_SIZE ticks
            if buf_len < BUFFER_SIZE:
                pad = np.zeros((BUFFER_SIZE - buf_len, 9), dtype=np.float32)
                snapshot = np.vstack([pad, arr])
            else:
                snapshot = arr

        self.snapshots.append((snapshot, brick.timestamp))

        # ── 3. Assemble Tensors ──────────────────────────
        if len(self.snapshots) < MACRO_HISTORY:
            return None  # Not enough brick history yet

        # Micro: stack last 10 snapshots → (10, 100, 9)
        micro_tensor = np.stack([s[0] for s in self.snapshots])

        # Macro: stack last 10 macro vectors → (10, 3)
        macro_list = list(self.macro_history)
        if len(macro_list) < MACRO_HISTORY:
            pad_count = MACRO_HISTORY - len(macro_list)
            pad = [np.zeros(3, dtype=np.float32)] * pad_count
            macro_list = pad + macro_list
        macro_tensor = np.stack(macro_list)

        # Add batch dimension: (1, 10, 100, 9) and (1, 10, 3)
        return (
            micro_tensor[np.newaxis, ...],
            macro_tensor[np.newaxis, ...]
        )
