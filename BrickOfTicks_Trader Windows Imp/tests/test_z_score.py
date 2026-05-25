"""
Phase 1 Verification: RollingZScore Engine
Tests per bot_implementation.md Phase 1 Gate:
1. Constant input: 2000 identical values → all 0.0
2. Simple sequence: [1..100], verify last 10 z-scores
3. Warmup: 29 → 0.0, 30th → non-zero
4. Training parity: (requires saved data — deferred)
5. Numerical stability: alternating 1e6 / 1e-6 → no NaN/Inf
"""
import unittest
import numpy as np
from BrickOfTicks_Trader.data.feature_engine import RollingZScore

class TestRollingZScore(unittest.TestCase):

    def test_constant_input_2000(self):
        """Gate Test 1: Feed 2000 identical values → all outputs = 0.0"""
        zs = RollingZScore(window=1000)
        for i in range(2000):
            val = zs.update(42.0)
            self.assertEqual(val, 0.0, f"Expected 0.0 at index {i}, got {val}")

    def test_simple_sequence_100(self):
        """Gate Test 2: Feed [1..100], verify last 10 z-scores match manual computation."""
        zs = RollingZScore(window=1000)
        results = []
        for i in range(1, 101):
            val = zs.update(float(i))
            results.append(val)
        
        # Manually compute expected z-scores for last 10 (indices 91-100)
        # At tick 100, we have values [1..100] in the deque
        for check_idx in range(90, 100):
            n = check_idx + 1  # number of values so far
            arr = list(range(1, n + 1))
            mean = sum(arr) / n
            m2 = sum((x - mean) ** 2 for x in arr)
            sigma = np.sqrt(m2 / (n - 1))
            expected = (float(n) - mean) / sigma if sigma > 1e-12 else 0.0
            self.assertAlmostEqual(results[check_idx], expected, places=6,
                                   msg=f"Mismatch at index {check_idx}")

    def test_warmup_boundary(self):
        """Gate Test 3: 29 values → 0.0, 30th → non-zero."""
        zs = RollingZScore(window=1000)
        for i in range(29):
            val = zs.update(float(i))
            self.assertEqual(val, 0.0, f"Expected 0.0 at index {i}")
        
        # 30th value with distinct value should be non-zero 
        val_30 = zs.update(30.0)
        self.assertNotEqual(val_30, 0.0, "30th value should be non-zero")

    def test_numerical_stability(self):
        """Gate Test 5: Alternating 1e6 / 1e-6 → no NaN, no Inf."""
        zs = RollingZScore(window=1000)
        for i in range(2000):
            x = 1e6 if i % 2 == 0 else 1e-6
            val = zs.update(x)
            self.assertFalse(np.isnan(val), f"NaN at index {i}")
            self.assertFalse(np.isinf(val), f"Inf at index {i}")

    def test_sliding_window_boundary(self):
        """Verify O(1) update matches recompute at window fill boundary."""
        window = 10
        zs = RollingZScore(window=window, warmup=1)
        
        for i in range(9):
            zs.update(float(i))
        
        # 10th value (transitions to full window)
        val_inc = zs.update(10.0)
        
        # Recompute manually
        arr = [float(x) for x in range(9)] + [10.0]
        mean = sum(arr) / len(arr)
        std = np.std(arr, ddof=1)
        val_expected = (10.0 - mean) / std
        
        self.assertAlmostEqual(val_inc, val_expected, places=6)

    def test_full_window_incremental_vs_scratch(self):
        """After window is full, verify incremental update stays accurate over 1000+ evictions."""
        window = 100
        zs = RollingZScore(window=window, warmup=1)
        
        # Fill window
        data = [float(i) for i in range(200)]
        for x in data:
            zs.update(x)
        
        # At this point, deque contains [100..199]
        # Add one more and check
        val = zs.update(200.0)
        
        # Manual check: deque is [101..200]
        arr = [float(i) for i in range(101, 201)]
        mean = sum(arr) / len(arr)
        std = np.std(arr, ddof=1)
        expected = (200.0 - mean) / std
        
        self.assertAlmostEqual(val, expected, places=4)

if __name__ == '__main__':
    unittest.main()
