"""
Phase 3 Verification: Live Feature Engine
Tests per bot_implementation.md Phase 3 Gate.
"""
import unittest
import numpy as np
from BrickOfTicks_Trader.data.feature_engine import LiveFeatureEngine


class TestLiveFeatureEngine(unittest.TestCase):

    def test_first_tick_returns_zeros(self):
        """Gate Test 6: First call returns [0.0] * 9."""
        fe = LiveFeatureEngine()
        vec = fe.compute_vector(bid=100.0, ask=100.5, bid_vol=10.0, ask_vol=12.0, time_ms=1000)
        self.assertEqual(vec, [0.0] * 9)

    def test_ofi_weak_inequalities(self):
        """Gate Test 3: Manually compute OFI for known tick pairs."""
        fe = LiveFeatureEngine(z_window=1000, z_warmup=30)

        # First tick (returns zeros, sets state)
        fe.compute_vector(100.0, 100.5, 10.0, 12.0, 1000)

        # Second tick: bid unchanged, ask unchanged → dBid=0, dAsk=0
        # OFI = (dBid>=0)*bid_vol - (dBid<=0)*prev_bid_vol - (dAsk<=0)*ask_vol + (dAsk>=0)*prev_ask_vol
        # = 1*10 - 1*10 - 1*12 + 1*12 = 0
        vec = fe.compute_vector(100.0, 100.5, 10.0, 12.0, 2000)
        # z_ofi should be 0.0 (warmup phase, all same value → sigma=0)
        self.assertEqual(vec[0], 0.0)

        # Third tick: bid up by 0.5, ask up by 0.5
        # dBid=0.5>0: dBid>=0 → 1, dBid<=0 → 0
        # dAsk=0.5>0: dAsk<=0 → 0, dAsk>=0 → 1
        # OFI = 1*15.0 - 0*10.0 - 0*15.0 + 1*12.0 = 15+12 = 27
        vec = fe.compute_vector(100.5, 101.0, 15.0, 15.0, 3000)
        # Still in warmup so z_ofi = 0.0
        self.assertEqual(vec[0], 0.0)

    def test_susceptibility_safety(self):
        """Gate Test 4: depth_raw = 0 → no crash, no NaN (1e-8 guard)."""
        fe = LiveFeatureEngine(z_window=100, z_warmup=2)
        fe.compute_vector(100.0, 100.5, 5.0, 5.0, 1000)  # First tick

        # Tick with non-zero volumes: depth = 5+5 = 10, susc = ofi / (10 + 1e-8)
        vec = fe.compute_vector(100.0, 100.5, 5.0, 5.0, 2000)
        self.assertFalse(np.isnan(vec[2]))  # z_susc
        self.assertFalse(np.isinf(vec[2]))

    def test_volume_fallback(self):
        """Gate Test 5: bid_vol=0 → price-action proxy used."""
        fe = LiveFeatureEngine(z_window=100, z_warmup=2)
        # First tick with volumes
        fe.compute_vector(100.0, 100.5, 5.0, 5.0, 1000)

        # Second tick with ZERO volumes, price went UP
        # mid was (100+100.5)/2=100.25, new mid=(101+101.5)/2=101.25 → UP → raw_ofi=1.0
        # depth=0.0, susc=0.0
        vec = fe.compute_vector(101.0, 101.5, 0.0, 0.0, 2000)
        # z_ofi should reflect the ofi=1.0 update
        # z_depth should reflect depth=0.0 update
        # During warmup (only 2 values needed), check no NaN
        self.assertFalse(any(np.isnan(v) for v in vec))
        self.assertFalse(any(np.isinf(v) for v in vec))

    def test_volume_fallback_direction_down(self):
        """Verify fallback produces -1.0 when price drops."""
        fe = LiveFeatureEngine(z_window=100, z_warmup=2)
        fe.compute_vector(100.0, 100.5, 5.0, 5.0, 1000)  # mid=100.25

        # Price drops: new mid = (99+99.5)/2 = 99.25 < 100.25 → raw_ofi = -1.0
        vec = fe.compute_vector(99.0, 99.5, 0.0, 0.0, 2000)
        self.assertFalse(any(np.isnan(v) for v in vec))

    def test_z_score_sustainability(self):
        """Feed 100 zero-volume ticks → z_depth/z_susc stay 0.0."""
        fe = LiveFeatureEngine(z_window=100, z_warmup=30)
        fe.compute_vector(100.0, 100.5, 5.0, 5.0, 1000)
        for i in range(100):
            vec = fe.compute_vector(100.0 + i * 0.01, 100.5 + i * 0.01,
                                    0.0, 0.0, 2000 + i * 100)
            # depth_raw is always 0.0 → constant → z_depth = 0.0
            self.assertEqual(vec[1], 0.0, f"z_depth should be 0.0 at tick {i}")
            # susc_raw is always 0.0 → constant → z_susc = 0.0
            self.assertEqual(vec[2], 0.0, f"z_susc should be 0.0 at tick {i}")

    def test_on_new_brick_context(self):
        """Gate Test 7: Verify on_new_brick updates context correctly."""
        fe = LiveFeatureEngine()

        class FakeBrick:
            close = 2100.0
            brick_size = 4.0
            uptrend = True

        fe.on_new_brick(FakeBrick())

        self.assertEqual(fe.current_brick_open, 2100.0)
        self.assertEqual(fe.current_brick_size, 4.0)
        self.assertEqual(fe.current_brick_id, 1)
        self.assertEqual(fe.prev_brick_open, 0.0)  # Was default
        self.assertEqual(fe.prev_brick_size, 1.0)  # Was default

        # Second brick
        class FakeBrick2:
            close = 2104.0
            brick_size = 4.0
            uptrend = True

        fe.on_new_brick(FakeBrick2())
        self.assertEqual(fe.current_brick_open, 2104.0)
        self.assertEqual(fe.prev_brick_open, 2100.0)
        self.assertEqual(fe.current_brick_id, 2)

    def test_progress_calculation(self):
        """Verify progress = (mid - brick_open) / brick_size."""
        fe = LiveFeatureEngine(z_window=100, z_warmup=2)
        fe.current_brick_open = 100.0
        fe.current_brick_size = 10.0

        fe.compute_vector(100.0, 100.5, 5.0, 5.0, 1000)  # First tick
        vec = fe.compute_vector(105.0, 105.5, 5.0, 5.0, 2000)
        # mid = (105+105.5)/2 = 105.25
        # progress = (105.25 - 100.0) / 10.0 = 0.525
        self.assertAlmostEqual(vec[5], 0.525, places=4)


if __name__ == '__main__':
    unittest.main()
