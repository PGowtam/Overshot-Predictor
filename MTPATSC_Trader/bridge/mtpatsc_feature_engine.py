"""
MTPATSC Trader — Live Feature Engine
======================================
Pure Python port of the C++ ANCS, Candle, and Momentum feature extractors.
Computes all features needed by the MTPATSC 5-class setup classifier at each
brick close, using the intra-brick tick data collected by the RenkoBuilder.

Feature dimensions:
  - ancs_fine:   (10, 6) = 60 floats  (10-segment OHLC + timing from intra-brick ticks)
  - ancs_coarse: (5, 6)  = 30 floats  (5-segment OHLC + timing)
  - candle_features: (15,) floats     (body, wicks, range, ratios)
  - momentum:   (19,) floats          (3-phase stats, acceleration, spreads, choppiness)
  - history:    (5, 5, 6) = 150 floats (rolling buffer of last 5 bricks' coarse ANCS)

Exact parity with:
  mtpatsc_engine.cpp :: compute_ancs(), compute_candle_features(), compute_momentum_features()
"""

import math
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class MTPatscFeatureEngine:
    """
    Stateful feature engine that computes MTPATSC features from live
    Renko bricks with their intra-brick tick data.
    """

    def __init__(self, scaler_path: str = None):
        """
        Args:
            scaler_path: Path to the fitted RobustScaler (scalar_scaler.pkl).
        """
        # Rolling history: stores the last 5 bricks' coarse ANCS (each 5x6 = 30 floats)
        self.rolling_history = deque(maxlen=5)
        # Initialize with zeros
        for _ in range(5):
            self.rolling_history.append(np.zeros(30, dtype=np.float32))

        # Load scaler
        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scalar scaler from {scaler_path}")
        else:
            logger.warning("No scaler path provided or file not found. Scalars will be unscaled.")

        self.brick_count = 0

    def on_brick_close(self, brick) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute all MTPATSC features for a newly closed brick.

        Args:
            brick: BrickEvent with intra_ticks list of {'bid', 'ask', 'time_msc'} dicts.

        Returns:
            Dict of tensors ready for model input, or None if insufficient history.
            Keys: 'ancs_fine', 'ancs_coarse', 'history', 'scalars'
        """
        ticks = brick.intra_ticks
        brick_size = brick.brick_size
        brick_open = brick.open
        direction = brick.uptrend  # 1 or -1

        # Compute features
        ancs_fine = self._compute_ancs(ticks, n_segments=10, brick_open_price=brick_open, brick_size=brick_size)
        ancs_coarse = self._compute_ancs(ticks, n_segments=5, brick_open_price=brick_open, brick_size=brick_size)
        candle_features = self._compute_candle_features(ticks, brick_size, direction)
        momentum = self._compute_momentum_features(ticks, brick_size)

        # Update rolling history (shift left, append new coarse)
        self.rolling_history.append(ancs_coarse.copy())
        self.brick_count += 1

        # Need at least 5 bricks of history for the history tensor
        if self.brick_count < 5:
            return None

        # Build history tensor: (5, 5, 6) from rolling_history
        # Each entry in rolling_history is 30 floats = 5 segments × 6 features
        history = np.zeros((5, 5, 6), dtype=np.float32)
        for i, coarse in enumerate(self.rolling_history):
            history[i] = coarse.reshape(5, 6)

        # Scale scalar features (candle=15 + momentum=19 = 34)
        scalars = np.concatenate([candle_features, momentum]).reshape(1, -1)
        if self.scaler is not None:
            scalars = self.scaler.transform(scalars).astype(np.float32)
        else:
            scalars = scalars.astype(np.float32)

        # Reshape for model input (batch dim = 1)
        return {
            'ancs_fine': ancs_fine.reshape(1, 10, 6),
            'ancs_coarse': ancs_coarse.reshape(1, 5, 6),
            'history': history.reshape(1, 5, 5, 6),
            'scalars': scalars,  # already (1, 34)
        }

    def _compute_ancs(self, ticks: List[Dict], n_segments: int,
                      brick_open_price: float, brick_size: float) -> np.ndarray:
        """
        Compute Anchored Normalised Core State (ANCS).
        Exact port of C++ compute_ancs().

        Returns: np.ndarray of shape (n_segments * 6,) = flat array
        """
        out = np.zeros(n_segments * 6, dtype=np.float32)
        N = len(ticks)

        if N == 0:
            return out

        seg_size = max(1, N // n_segments)
        brick_start_time = ticks[0]['time_msc']
        brick_end_time = ticks[-1]['time_msc']
        brick_duration = max(1.0, float(brick_end_time - brick_start_time))

        for i in range(n_segments):
            start = i * seg_size
            end = min((i + 1) * seg_size, N) if i < n_segments - 1 else N

            if start >= N or end <= start:
                continue

            seg_open = ticks[start]['bid']
            seg_close = ticks[end - 1]['bid']
            seg_high = seg_open
            seg_low = seg_open

            for k in range(start, end):
                p = ticks[k]['bid']
                if p > seg_high:
                    seg_high = p
                if p < seg_low:
                    seg_low = p

            out[i * 6 + 0] = float((seg_open - brick_open_price) / brick_size)
            out[i * 6 + 1] = float((seg_high - brick_open_price) / brick_size)
            out[i * 6 + 2] = float((seg_low - brick_open_price) / brick_size)
            out[i * 6 + 3] = float((seg_close - brick_open_price) / brick_size)

            duration_frac = float(ticks[end - 1]['time_msc'] - brick_start_time) / brick_duration
            tick_frac = float(end - start) / N
            out[i * 6 + 4] = float(duration_frac)
            out[i * 6 + 5] = float(tick_frac)

        return out

    def _compute_candle_features(self, ticks: List[Dict],
                                  brick_size: float, direction: int) -> np.ndarray:
        """
        Compute 15 candle-structure features.
        Exact port of C++ compute_candle_features().
        """
        out = np.zeros(15, dtype=np.float32)

        if not ticks:
            return out

        O = ticks[0]['bid']
        C = ticks[-1]['bid']
        H = O
        L = O
        for t in ticks:
            p = t['bid']
            if p > H:
                H = p
            if p < L:
                L = p

        body = abs(C - O)
        max_OC = max(O, C)
        min_OC = min(O, C)
        upper_wick = H - max_OC
        lower_wick = min_OC - L
        full_range = H - L

        out[0] = float(body / brick_size)
        out[1] = float(upper_wick / brick_size)
        out[2] = float(lower_wick / brick_size)
        out[3] = float(full_range / brick_size)
        out[4] = float((C - O) / brick_size)
        out[5] = float(upper_wick / (full_range + 1e-8))
        out[6] = float(lower_wick / (full_range + 1e-8))
        out[7] = float(body / (full_range + 1e-8))
        out[8] = float((C - L) / (full_range + 1e-8))
        out[9] = float((O - L) / (full_range + 1e-8))
        out[10] = 1.0 if C > (H + L) / 2.0 else 0.0
        if direction == 1:
            out[11] = float((H - C) / brick_size)
        else:
            out[11] = float((C - L) / brick_size)
        out[12] = float(min(full_range / brick_size, 3.0))
        out[13] = float(abs(O - L) / brick_size)
        out[14] = float(abs(H - O) / brick_size)

        return out

    def _compute_momentum_features(self, ticks: List[Dict],
                                    brick_size: float) -> np.ndarray:
        """
        Compute 19 momentum features.
        Exact port of C++ compute_momentum_features().
        """
        out = np.zeros(19, dtype=np.float32)
        N = len(ticks)

        if N == 0:
            return out

        def phase_stats(start: int, end: int, out_idx: int):
            if end - start < 2:
                return
            p0 = ticks[start]['bid']
            p1 = ticks[end - 1]['bid']
            H = p0
            L = p0
            s = 0.0
            sq_s = 0.0
            for k in range(start, end):
                p = ticks[k]['bid']
                if p > H:
                    H = p
                if p < L:
                    L = p
                s += p
                sq_s += p * p
            count = end - start
            mean = s / count
            var = (sq_s / count) - (mean * mean)
            std_dev = math.sqrt(var) if var > 0 else 0.0

            out[out_idx] = float((p1 - p0) / brick_size)
            out[out_idx + 1] = float((H - p0) / brick_size)
            out[out_idx + 2] = float((L - p0) / brick_size)
            out[out_idx + 3] = float(std_dev / brick_size)

        phase_stats(0, N // 3, 0)
        phase_stats(N // 3, 2 * N // 3, 4)
        phase_stats(2 * N // 3, N, 8)

        # Acceleration
        early_move = abs(float(out[0]))
        late_move = abs(float(out[8]))
        out[12] = float(late_move - early_move)

        # Velocity ratio
        velocity_ratio = 1.0
        if N > 10:
            early_count = N // 3
            late_count = N - 2 * N // 3
            if early_count > 1:
                dt_early = float(ticks[early_count - 1]['time_msc'] - ticks[0]['time_msc']) / (early_count - 1)
            else:
                dt_early = 1000.0
            if late_count > 1:
                dt_late = float(ticks[-1]['time_msc'] - ticks[2 * N // 3]['time_msc']) / (late_count - 1)
            else:
                dt_late = 1000.0
            velocity_ratio = dt_early / (dt_late + 1e-3)
        out[13] = float(velocity_ratio)

        # Spread features
        n_open = max(1, N // 10)
        spread_open = sum(t['ask'] - t['bid'] for t in ticks[:n_open]) / n_open

        start_close = max(0, 9 * N // 10)
        n_close = N - start_close
        spread_close = sum(t['ask'] - t['bid'] for t in ticks[start_close:]) / max(1, n_close)

        out[14] = float((spread_close - spread_open) / (spread_open + 1e-8))

        # Choppiness (direction changes)
        dir_changes = 0
        for i in range(1, N - 1):
            d1 = ticks[i]['bid'] - ticks[i - 1]['bid']
            d2 = ticks[i + 1]['bid'] - ticks[i]['bid']
            if d1 * d2 < 0:
                dir_changes += 1
        out[15] = float(dir_changes / max(1, N))

        # Duration
        duration_ms = ticks[-1]['time_msc'] - ticks[0]['time_msc']
        out[16] = float(math.log1p(duration_ms / 1000.0))

        # Spread normalized
        out[17] = float(spread_open / brick_size)
        out[18] = float(spread_close / brick_size)

        return out
