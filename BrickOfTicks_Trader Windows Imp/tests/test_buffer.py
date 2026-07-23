"""
Phase 4 Verification: Micro-Buffer & Tensor Assembly
Tests per bot_implementation.md Phase 4 Gate.
"""
import unittest
import numpy as np
from math import log
from collections import namedtuple
from BrickOfTicks_Trader.inference.buffer import InferenceBuffer, BUFFER_SIZE, MACRO_HISTORY


# Lightweight fake brick for testing
FakeBrick = namedtuple('FakeBrick', ['open', 'close', 'high', 'low',
                                      'uptrend', 'timestamp', 'brick_size', 'sequence'])


class TestInferenceBuffer(unittest.TestCase):

    def _make_brick(self, close=100.0, uptrend=True, ts=1000, brick_size=10.0):
        return FakeBrick(
            open=close - brick_size if uptrend else close + brick_size,
            close=close, high=close, low=close - brick_size,
            uptrend=uptrend, timestamp=ts, brick_size=brick_size, sequence="1"
        )

    def test_shape(self):
        """Gate Test 1: 15 brick closes → first 9 return None, 10th returns correct shapes."""
        buf = InferenceBuffer()

        for brick_num in range(15):
            # Add some ticks for each brick
            for t in range(20):
                vec = [float(t)] * 9
                buf.append_tick(vec, brick_num)

            brick = self._make_brick(
                close=100.0 + brick_num * 10,
                ts=1000 * (brick_num + 1),
                uptrend=True
            )
            result = buf.on_brick_close(brick)

            if brick_num < 9:
                self.assertIsNone(result, f"Expected None at brick {brick_num}")
            else:
                self.assertIsNotNone(result, f"Expected tensors at brick {brick_num}")
                micro, macro = result
                self.assertEqual(micro.shape, (1, 10, 100, 9),
                                 f"micro shape mismatch at brick {brick_num}")
                self.assertEqual(macro.shape, (1, 10, 3),
                                 f"macro shape mismatch at brick {brick_num}")

    def test_zero_padding(self):
        """Gate Test 2: 30 ticks only → first 70 rows zeros, last 30 non-zero."""
        buf = InferenceBuffer()

        # Add 30 ticks
        for t in range(30):
            vec = [1.0] * 9
            buf.append_tick(vec, 0)

        brick = self._make_brick(ts=1000)

        # Close 10 bricks to get a result (first 9 return None)
        buf.on_brick_close(brick)  # brick 1
        for i in range(9):
            # Add some ticks and close more bricks
            for t in range(5):
                buf.append_tick([float(t)] * 9, i + 1)
            brick = self._make_brick(ts=2000 + i * 1000)
            result = buf.on_brick_close(brick)

        self.assertIsNotNone(result)
        micro, macro = result

        # Check the FIRST snapshot (index 0) which had only 30 ticks
        first_snapshot = micro[0, 0]  # (100, 9)
        # First 70 rows should be zeros
        np.testing.assert_array_equal(first_snapshot[:70], np.zeros((70, 9)))
        # Last 30 rows should be non-zero (they were all 1.0)
        self.assertTrue(np.any(first_snapshot[70:] != 0))

    def test_flag_curr_rewrite(self):
        """Gate Test 3: 50 ticks id=0 + 50 ticks id=1 → correct rewrite."""
        buf = InferenceBuffer()

        # 50 ticks brick_id=0
        for t in range(50):
            vec = [0.0] * 9
            vec[6] = 1.0  # Flag_Curr initially set to 1
            buf.append_tick(vec, 0)

        # 50 ticks brick_id=1
        for t in range(50):
            vec = [0.0] * 9
            vec[6] = 1.0
            buf.append_tick(vec, 1)

        # Close brick 1 (current_brick_id becomes 1)
        brick = self._make_brick(ts=1000)
        buf.on_brick_close(brick)

        # The snapshot should have:
        # - First 50 ticks (bid=0): Flag_Curr = 0 (not current brick)
        # - Last 50 ticks (bid=1): Flag_Curr = 1 (current brick)
        snapshot = buf.snapshots[-1][0]  # (100, 9)
        for i in range(50):
            self.assertEqual(snapshot[i, 6], 0.0,
                             f"Row {i} should have Flag_Curr=0")
        for i in range(50, 100):
            self.assertEqual(snapshot[i, 6], 1.0,
                             f"Row {i} should have Flag_Curr=1")

    def test_decay_rewrite(self):
        """Gate Test 4: Verify Decay = (current_id - tick_id) / BUFFER_SIZE."""
        buf = InferenceBuffer()

        # 60 ticks brick_id=0
        for t in range(60):
            buf.append_tick([0.0] * 9, 0)

        # 40 ticks brick_id=1
        for t in range(40):
            buf.append_tick([0.0] * 9, 1)

        # Close brick → current_brick_id = 1
        brick = self._make_brick(ts=1000)
        buf.on_brick_close(brick)

        snapshot = buf.snapshots[-1][0]  # (100, 9)
        # First 60 ticks (id=0): Decay = (1-0)/100 = 0.01
        for i in range(60):
            self.assertAlmostEqual(snapshot[i, 8], 0.01, places=6,
                                   msg=f"Row {i} decay mismatch")
        # Last 40 ticks (id=1): Decay = (1-1)/100 = 0.0
        for i in range(60, 100):
            self.assertAlmostEqual(snapshot[i, 8], 0.0, places=6,
                                   msg=f"Row {i} decay mismatch")

    def test_continuity(self):
        """Gate Test 5: After brick close, old ticks still present (shifted but present)."""
        buf = InferenceBuffer()

        # Add 80 ticks with brick_id=0
        for t in range(80):
            buf.append_tick([float(t)] * 9, 0)

        brick = self._make_brick(ts=1000)
        buf.on_brick_close(brick)

        # The buffer still has 80 ticks
        self.assertEqual(len(buf.micro_buffer), 80)

        # Add 30 more ticks with brick_id=1
        for t in range(30):
            buf.append_tick([100.0 + t] * 9, 1)

        # Buffer now has 100 ticks (80 from id=0, 20 from id=1... wait, deque maxlen=100)
        # Actually 80+30=110, but maxlen=100, so oldest 10 are dropped
        # Buffer has 70 from id=0 and 30 from id=1
        self.assertEqual(len(buf.micro_buffer), 100)

        # Old ticks from id=0 are still in the buffer
        old_tick_count = sum(1 for _, bid in buf.micro_buffer if bid == 0)
        self.assertEqual(old_tick_count, 70)  # 80-10=70 remain

    def test_macro_vector(self):
        """Gate Test 6: Known durations/sizes → verify log_dur, direction, z_size."""
        buf = InferenceBuffer()

        # First brick close
        for t in range(10):
            buf.append_tick([0.0] * 9, 0)

        b1 = self._make_brick(close=100.0, uptrend=True, ts=5000, brick_size=10.0)
        buf.on_brick_close(b1)

        # Second brick close (3 seconds later)
        for t in range(10):
            buf.append_tick([0.0] * 9, 1)

        b2 = self._make_brick(close=90.0, uptrend=False, ts=8000, brick_size=10.0)
        buf.on_brick_close(b2)

        # Check macro_history
        self.assertEqual(len(buf.macro_history), 2)

        # First macro: duration = 0 (first brick, no previous)
        m0 = buf.macro_history[0]
        self.assertAlmostEqual(m0[0], log(0 + 1), places=4)  # log_dur
        self.assertAlmostEqual(m0[1], 1.0, places=4)  # direction (UP)

        # Second macro: duration = (8000-5000)/1000 = 3.0s
        m1 = buf.macro_history[1]
        self.assertAlmostEqual(m1[0], log(3.0 + 1), places=4)  # log_dur
        self.assertAlmostEqual(m1[1], -1.0, places=4)  # direction (DOWN)


if __name__ == '__main__':
    unittest.main()
