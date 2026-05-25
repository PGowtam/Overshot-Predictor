"""
BrickOfTicks Socket Bridge — Feature Engine
===========================================
Computes the 9D tick feature vector and 3D macro-vector for every brick.
Includes Volume Fallback for retail brokers with missing tick volume.
"""

import math
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

class RollingZScore:
    """O(1) incremental z-score with sliding window.
    
    Uses Welford-like incremental formula for full window.
    Returns 0.0 when window has fewer than 30 values or variance is 0.
    """
    def __init__(self, window: int = 1000):
        self.window = window
        self.deque = deque(maxlen=window)
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x_new: float) -> float:
        N = len(self.deque)

        if N == self.window:
            # Full window — O(1) incremental update
            x_old = self.deque[0]
            self.deque.append(x_new)
            mean_new = self.mean + (x_new - x_old) / self.window
            self.M2 = self.M2 + (x_new - x_old) * ((x_new - mean_new) + (x_old - self.mean))
            self.mean = mean_new
            if self.M2 < 0:
                self.M2 = 0.0
            sigma = math.sqrt(self.M2 / (self.window - 1)) if self.window > 1 else 0.0
            if sigma < 1e-12:
                return 0.0
            return (x_new - self.mean) / sigma
        else:
            # Filling phase
            self.deque.append(x_new)
            N = len(self.deque)
            if N < 30:
                return 0.0  # Warmup
            # Recompute from scratch to avoid precision issues in filling phase
            arr = list(self.deque)
            self.mean = sum(arr) / N
            self.M2 = sum((x - self.mean) ** 2 for x in arr)
            sigma = math.sqrt(self.M2 / (N - 1)) if N > 1 else 0.0
            if sigma < 1e-12:
                return 0.0
            return (x_new - self.mean) / sigma


class LiveFeatureEngine:
    def __init__(self):
        self.zs_ofi = RollingZScore(1000)
        self.zs_depth = RollingZScore(1000)
        self.zs_susc = RollingZScore(1000)
        self.zs_vel = RollingZScore(1000)
        self.zs_spread = RollingZScore(1000)

        self.prev_bid = None
        self.prev_ask = None
        self.prev_bid_vol = None
        self.prev_ask_vol = None
        self.prev_time_ms = None

        self.brick_open = None
        self.brick_size = 1.0
        self.prev_brick_open = None
        self.prev_brick_size = 1.0
        
        self.brick_size_history = []
        self.last_brick_time = None

    def _init_prev(self, bid, ask, bid_vol, ask_vol, time_ms):
        self.prev_bid = bid
        self.prev_ask = ask
        self.prev_bid_vol = bid_vol
        self.prev_ask_vol = ask_vol
        self.prev_time_ms = time_ms

    def update_brick_size(self, new_brick_size: float):
        """Update brick size for Progress calculation after daily rollover."""
        self.brick_size = new_brick_size

    def on_new_brick(self, brick_event):
        """Called by main orchestrator when RenkoBuilder emits a new brick."""
        self.prev_brick_open = self.brick_open
        self.prev_brick_size = self.brick_size
        
        self.brick_open = brick_event.open
        self.brick_size = brick_event.brick_size
        
        self.brick_size_history.append(brick_event.brick_size)
        
        duration = 0.0
        if self.last_brick_time is not None:
            duration = (brick_event.timestamp - self.last_brick_time) / 1000.0
            if duration < 0:
                duration = 0.0  # Safety: out-of-order timestamps from history replay
        self.last_brick_time = brick_event.timestamp
        
        # Macro vector computation
        log_dur = math.log(duration + 1)
        direction = 1.0 if brick_event.uptrend == 1 else -1.0
        
        z_size = 0.0
        if len(self.brick_size_history) >= 2:
            recent = self.brick_size_history[-50:]
            mu = np.mean(recent)
            sigma = np.std(recent, ddof=1) if len(recent) > 1 else 0.0
            if sigma >= 1e-12:
                z_size = (brick_event.brick_size - mu) / sigma
                
        self.last_macro = [log_dur, direction, float(z_size)]

    def compute_vector(self, bid: float, ask: float, bid_vol: float, ask_vol: float, time_ms: int) -> list:
        """
        Compute the 9D feature vector for a new tick.
        """
        mid = (bid + ask) / 2.0
        
        if self.prev_bid is None:
            self._init_prev(bid, ask, bid_vol, ask_vol, time_ms)
            if self.brick_open is None:
                self.brick_open = mid # Fallback
            return [0.0] * 9

        # FR-PY-FEAT-04: Volume Fallback
        if bid_vol <= 0 or ask_vol <= 0:
            prev_mid = (self.prev_bid + self.prev_ask) / 2.0
            if mid > prev_mid:
                raw_ofi = 1.0
            elif mid < prev_mid:
                raw_ofi = -1.0
            else:
                raw_ofi = 0.0
            depth_raw = 0.0
            susc_raw = 0.0
        else:
            # FR-PY-FEAT-02: Weak inequalities
            dBid = bid - self.prev_bid
            dAsk = ask - self.prev_ask
            raw_ofi = (
                (1 if dBid >= 0 else 0) * bid_vol
              - (1 if dBid <= 0 else 0) * self.prev_bid_vol
              - (1 if dAsk <= 0 else 0) * ask_vol
              + (1 if dAsk >= 0 else 0) * self.prev_ask_vol
            )
            depth_raw = bid_vol + ask_vol
            # FR-PY-FEAT-03: Raw division FIRST
            susc_raw = raw_ofi / (depth_raw + 1e-8)

        # Velocity
        dt = (time_ms - self.prev_time_ms) / 1000.0
        vel_raw = 1.0 / (dt + 1e-3)
        
        # Spread
        spread_raw = ask - bid

        # Z-scoring
        z_ofi = self.zs_ofi.update(raw_ofi)
        z_depth = self.zs_depth.update(depth_raw)
        z_susc = self.zs_susc.update(susc_raw)
        z_vel = self.zs_vel.update(vel_raw)
        z_spread = self.zs_spread.update(spread_raw)

        # Progress
        if self.brick_open is None:
            self.brick_open = mid
        progress = (mid - self.brick_open) / self.brick_size

        # Flag_Curr (Always 1 for streaming, managed by Buffer)
        flag_curr = 1.0

        # Flag_Zone
        flag_zone = 0.0
        if self.prev_brick_open is not None:
            if abs(mid - self.prev_brick_open) >= self.prev_brick_size:
                flag_zone = 1.0
                
        # Decay (Managed by Buffer)
        decay = 0.0

        # Update prevs
        self._init_prev(bid, ask, bid_vol, ask_vol, time_ms)

        return [z_ofi, z_depth, z_susc, z_vel, z_spread, progress, flag_curr, flag_zone, decay]
