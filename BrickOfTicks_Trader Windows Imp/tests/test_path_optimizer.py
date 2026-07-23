"""
Phase 13: Tests for PathOptimizer

Tests cover:
1. Day boundary detection from tick timestamps
2. Candidate price generation from anchor day range
3. Renko builder parity with training pipeline
4. Trade simulation parity with training pipeline
5. Optimal anchor selection with synthetic data
6. Edge cases (single day, empty data, weekend gaps)
"""

import unittest
import numpy as np
from datetime import datetime, timezone, timedelta

from BrickOfTicks_Trader.data.path_optimizer import PathOptimizer


def _make_ticks(prices, start_ts_msc, interval_msc=100):
    """
    Helper: Create a fake structured numpy array mimicking MT5 ticks.
    """
    n = len(prices)
    dtype = np.dtype([
        ('bid', np.float64),
        ('ask', np.float64),
        ('last', np.float64),
        ('volume', np.uint64),
        ('time_msc', np.int64),
        ('flags', np.uint32),
        ('volume_real', np.float64),
    ])
    ticks = np.zeros(n, dtype=dtype)
    for i, p in enumerate(prices):
        ticks[i]['bid'] = p
        ticks[i]['ask'] = p + 0.22  # Typical spread
        ticks[i]['time_msc'] = start_ts_msc + i * interval_msc
        ticks[i]['volume_real'] = 0.0
    return ticks


def _make_multiday_ticks(day_prices_map, base_date=None):
    """
    Helper: Create ticks spanning multiple UTC days.
    
    Args:
        day_prices_map: list of (date_offset_days, [price1, price2, ...])
        base_date: starting date (default: 2026-03-25, a Wednesday)
    
    Returns numpy structured array.
    """
    if base_date is None:
        base_date = datetime(2026, 3, 25, tzinfo=timezone.utc)
    
    all_ticks = []
    dtype = np.dtype([
        ('bid', np.float64),
        ('ask', np.float64),
        ('last', np.float64),
        ('volume', np.uint64),
        ('time_msc', np.int64),
        ('flags', np.uint32),
        ('volume_real', np.float64),
    ])
    
    for day_offset, prices in day_prices_map:
        day_start = base_date + timedelta(days=day_offset)
        day_start_msc = int(day_start.timestamp() * 1000)
        
        for i, p in enumerate(prices):
            tick = np.zeros(1, dtype=dtype)
            # Space ticks 1 second apart within the day
            tick[0]['bid'] = p
            tick[0]['ask'] = p + 0.22
            tick[0]['time_msc'] = day_start_msc + i * 1000
            tick[0]['volume_real'] = 0.0
            all_ticks.append(tick[0])
    
    result = np.array(all_ticks, dtype=dtype)
    return result


class TestDayBoundaries(unittest.TestCase):
    """Test _find_day_boundaries correctly identifies UTC day transitions."""

    def setUp(self):
        self.optimizer = PathOptimizer()

    def test_single_day(self):
        """Single day of ticks should produce exactly 1 boundary."""
        # 2026-03-25 00:00:00 UTC in msc
        base_msc = int(datetime(2026, 3, 25, tzinfo=timezone.utc).timestamp() * 1000)
        ts_array = np.array([base_msc + i * 1000 for i in range(100)])  # 100 ticks, 1sec apart
        
        boundaries = self.optimizer._find_day_boundaries(ts_array)
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0][0], 0)    # start_idx
        self.assertEqual(boundaries[0][1], 100)  # end_idx

    def test_multi_day(self):
        """Ticks spanning 3 days should produce 3 boundaries."""
        boundaries_data = []
        for day in range(3):
            base_msc = int(datetime(2026, 3, 25 + day, tzinfo=timezone.utc).timestamp() * 1000)
            for i in range(50):
                boundaries_data.append(base_msc + i * 1000)
        
        ts_array = np.array(boundaries_data)
        boundaries = self.optimizer._find_day_boundaries(ts_array)
        self.assertEqual(len(boundaries), 3)
        
    def test_empty_array(self):
        """Empty array should return empty list."""
        boundaries = self.optimizer._find_day_boundaries(np.array([]))
        self.assertEqual(len(boundaries), 0)


class TestRenkoParity(unittest.TestCase):
    """Test _build_renko_from produces correct bricks matching training pipeline."""

    def setUp(self):
        self.optimizer = PathOptimizer()

    def test_uptrend_continuation(self):
        """Steadily rising prices should produce UP bricks."""
        brick_size = 10.0
        start_price = 100.0
        
        # Create prices that rise by brick_size increments
        prices = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0])
        timestamps = np.array([1000 * i for i in range(len(prices))])
        
        dates, closes, uptrends = self.optimizer._build_renko_from(
            timestamps, prices, 0, start_price, brick_size, window_start_ts=0
        )
        
        # With brick_size=10, from 100 we expect bricks at 110, 120, 130
        self.assertEqual(len(dates), 3)
        self.assertAlmostEqual(closes[0], 110.0)
        self.assertAlmostEqual(closes[1], 120.0)
        self.assertAlmostEqual(closes[2], 130.0)
        self.assertTrue(all(u == 1 for u in uptrends))

    def test_downtrend_continuation(self):
        """Steadily falling prices should produce DOWN bricks."""
        brick_size = 10.0
        start_price = 130.0
        
        # Start high, drop continuously
        prices = np.array([130.0, 125.0, 120.0, 115.0, 110.0, 105.0, 100.0])
        timestamps = np.array([1000 * i for i in range(len(prices))])
        
        dates, closes, uptrends = self.optimizer._build_renko_from(
            timestamps, prices, 0, start_price, brick_size, window_start_ts=0
        )
        
        # From 130, bricks at 120, 110, 100
        self.assertEqual(len(dates), 3)
        self.assertAlmostEqual(closes[0], 120.0)
        self.assertAlmostEqual(closes[1], 110.0)
        self.assertAlmostEqual(closes[2], 100.0)
        self.assertTrue(all(u == -1 for u in uptrends))

    def test_reversal_requires_2x(self):
        """Reversal from UP requires price to drop by 2x brick_size."""
        brick_size = 10.0
        start_price = 100.0
        
        # Go up first, then try a small drop (should NOT reverse), then big drop
        prices = np.array([
            100.0, 110.0,   # UP brick formed at 110
            105.0,           # Drop by 5 — NOT enough for reversal (needs 2x = 20)
            108.0,           # Still not enough
            90.0,            # Drop from 110 by 20 — NOW reversal triggers
        ])
        timestamps = np.array([1000 * i for i in range(len(prices))])
        
        dates, closes, uptrends = self.optimizer._build_renko_from(
            timestamps, prices, 0, start_price, brick_size, window_start_ts=0
        )
        
        # First brick: UP to 110
        # Then reversal: current_price goes to 110 - 20 = 90 → DOWN brick
        self.assertEqual(len(dates), 2)
        self.assertEqual(uptrends[0], 1)   # UP
        self.assertEqual(uptrends[1], -1)  # DOWN (reversal)
        self.assertAlmostEqual(closes[0], 110.0)
        self.assertAlmostEqual(closes[1], 90.0)

    def test_window_start_ts_filter(self):
        """Bricks before window_start_ts should be excluded from output."""
        brick_size = 10.0
        start_price = 100.0
        
        # Prices go from 100 to 140 in steps, each at a different timestamp
        prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        timestamps = np.array([1000, 2000, 3000, 4000, 5000])
        
        # Only include bricks from ts >= 3500
        # Bricks form at: ts=2000 (close=110), ts=3000 (close=120), ts=4000 (close=130), ts=5000 (close=140)
        # Only ts=4000 and ts=5000 are >= 3500
        dates, closes, uptrends = self.optimizer._build_renko_from(
            timestamps, prices, 0, start_price, brick_size, window_start_ts=3500
        )
        
        self.assertEqual(len(dates), 2)
        self.assertAlmostEqual(closes[0], 130.0)
        self.assertAlmostEqual(closes[1], 140.0)

    def test_neutral_state_first_move(self):
        """From neutral state, first move in either direction uses 1x threshold."""
        brick_size = 10.0
        start_price = 100.0
        
        # First move DOWN from neutral
        prices = np.array([100.0, 95.0, 90.0])
        timestamps = np.array([1000 * i for i in range(len(prices))])
        
        dates, closes, uptrends = self.optimizer._build_renko_from(
            timestamps, prices, 0, start_price, brick_size, window_start_ts=0
        )
        
        self.assertEqual(len(dates), 1)
        self.assertEqual(uptrends[0], -1)
        self.assertAlmostEqual(closes[0], 90.0)


class TestTradeSimulation(unittest.TestCase):
    """Test _simulate_profit matches training pipeline logic."""

    def setUp(self):
        self.optimizer = PathOptimizer()

    def test_tp_hit_buy(self):
        """BUY trade hitting TP should return +0.5."""
        brick_size = 10.0
        # Brick: UP close at 110, with a "next brick" to define end_time
        b_dates = np.array([1000, 5000])
        b_closes = np.array([110.0, 120.0])
        b_uptrends = np.array([1, 1])
        
        # TP = 110 + 10 = 120. Ticks after brick time go to 120.
        tick_ts = np.array([1500, 2000, 2500])
        tick_prices = np.array([112.0, 115.0, 120.0])
        
        pnl = self.optimizer._simulate_profit(
            b_dates, b_closes, b_uptrends, tick_ts, tick_prices, brick_size
        )
        # First brick: TP hit → +0.5, second brick: no ticks after 5000 → 0
        self.assertAlmostEqual(pnl, 0.5)

    def test_sl_hit_buy(self):
        """BUY trade hitting SL should return -0.5."""
        brick_size = 10.0
        b_dates = np.array([1000, 5000])
        b_closes = np.array([110.0, 120.0])
        b_uptrends = np.array([1, 1])
        
        # SL = 110 - 10 = 100. Ticks drop to 100.
        tick_ts = np.array([1500, 2000, 2500])
        tick_prices = np.array([108.0, 105.0, 100.0])
        
        pnl = self.optimizer._simulate_profit(
            b_dates, b_closes, b_uptrends, tick_ts, tick_prices, brick_size
        )
        self.assertAlmostEqual(pnl, -0.5)

    def test_tp_hit_sell(self):
        """SELL trade hitting TP should return +0.5."""
        brick_size = 10.0
        b_dates = np.array([1000, 5000])
        b_closes = np.array([110.0, 100.0])
        b_uptrends = np.array([-1, -1])
        
        # TP = 110 - 10 = 100. Ticks drop to 100.
        tick_ts = np.array([1500, 2000, 2500])
        tick_prices = np.array([108.0, 105.0, 100.0])
        
        pnl = self.optimizer._simulate_profit(
            b_dates, b_closes, b_uptrends, tick_ts, tick_prices, brick_size
        )
        self.assertAlmostEqual(pnl, 0.5)

    def test_breakeven_logic(self):
        """After BE trigger, SL moves to entry. Hitting SL should give 0 PnL."""
        brick_size = 10.0
        b_dates = np.array([1000])
        b_closes = np.array([110.0])
        b_uptrends = np.array([1])
        
        # BE trigger = 110 + 0.3125 * 10 = 113.125
        # Ticks: go up past BE trigger, then back down to entry
        tick_ts = np.array([1500, 2000, 2500, 3000])
        tick_prices = np.array([112.0, 114.0, 112.0, 110.0])
        
        pnl = self.optimizer._simulate_profit(
            b_dates, b_closes, b_uptrends, tick_ts, tick_prices, brick_size
        )
        # BE was triggered at 114.0 (>= 113.125)
        # Then price dropped to 110.0 (entry) — SL at entry → outcome = 0
        self.assertAlmostEqual(pnl, 0.0)

    def test_gap_continuation_instant_win(self):
        """Same-timestamp bricks with same direction give +1.0."""
        brick_size = 10.0
        # Two UP bricks at same timestamp
        b_dates = np.array([1000, 1000])
        b_closes = np.array([110.0, 120.0])
        b_uptrends = np.array([1, 1])
        
        tick_ts = np.array([1500, 2000])
        tick_prices = np.array([121.0, 122.0])
        
        pnl = self.optimizer._simulate_profit(
            b_dates, b_closes, b_uptrends, tick_ts, tick_prices, brick_size
        )
        # First brick: gap continuation → +1.0
        # Second brick: trades normally
        self.assertGreaterEqual(pnl, 1.0)

    def test_gap_reversal_instant_loss(self):
        """Same-timestamp bricks with opposite direction give -1.0."""
        brick_size = 10.0
        # UP brick followed by DOWN brick at same timestamp, then a third to end
        b_dates = np.array([1000, 1000, 5000])
        b_closes = np.array([110.0, 100.0, 90.0])
        b_uptrends = np.array([1, -1, -1])
        
        tick_ts = np.array([1500, 2000])
        tick_prices = np.array([95.0, 90.0])
        
        pnl = self.optimizer._simulate_profit(
            b_dates, b_closes, b_uptrends, tick_ts, tick_prices, brick_size
        )
        # First brick: gap reversal → -1.0
        # Second brick: SELL close=100, TP=90, ticks reach 90 → +0.5
        # Third brick: no ticks after 5000 → 0
        self.assertAlmostEqual(pnl, -0.5)

    def test_empty_bricks(self):
        """No bricks should return 0.0 PnL."""
        pnl = self.optimizer._simulate_profit(
            np.array([]), np.array([]), np.array([]),
            np.array([1000]), np.array([100.0]), 10.0
        )
        self.assertAlmostEqual(pnl, 0.0)


class TestOptimalAnchorSelection(unittest.TestCase):
    """Test find_optimal_anchor selects the best candidate."""

    def setUp(self):
        self.optimizer = PathOptimizer()

    def test_selects_profitable_anchor(self):
        """Given known price patterns, optimizer should find a non-None anchor."""
        # Create 3 days of data with a clear uptrend
        # Day 0 (anchor): 4700-4720 range
        # Day 1 (sim): steady rise 4710 → 4730
        # Day 2 (target/today): 4730+
        
        day0_prices = list(np.linspace(4700, 4720, 200))
        day1_prices = list(np.linspace(4710, 4750, 300))
        day2_prices = list(np.linspace(4740, 4760, 100))
        
        ticks = _make_multiday_ticks([
            (0, day0_prices),
            (1, day1_prices),
            (2, day2_prices),
        ])
        
        brick_size = 5.0  # Small brick for more bricks from this data
        
        best_price, best_idx, best_profit = self.optimizer.find_optimal_anchor(
            ticks, brick_size
        )
        
        self.assertIsNotNone(best_price)
        self.assertGreaterEqual(best_price, 4700)
        self.assertLessEqual(best_price, 4720)

    def test_single_day_fallback(self):
        """With only 1 day of data, should return first tick price."""
        prices = list(np.linspace(4700, 4720, 100))
        ticks = _make_multiday_ticks([(0, prices)])
        
        best_price, best_idx, best_profit = self.optimizer.find_optimal_anchor(
            ticks, 5.0
        )
        
        self.assertIsNotNone(best_price)
        self.assertAlmostEqual(best_price, 4700.0, places=0)

    def test_empty_ticks(self):
        """Empty tick array should return None."""
        best_price, best_idx, best_profit = self.optimizer.find_optimal_anchor(
            np.array([]), 5.0
        )
        self.assertIsNone(best_price)

    def test_none_ticks(self):
        """None tick array should return None."""
        best_price, best_idx, best_profit = self.optimizer.find_optimal_anchor(
            None, 5.0
        )
        self.assertIsNone(best_price)


class TestCandidateGeneration(unittest.TestCase):
    """Test that candidate prices are generated correctly."""

    def setUp(self):
        self.optimizer = PathOptimizer()

    def test_candidates_within_anchor_range(self):
        """All candidates should be within anchor_low to anchor_high."""
        # We test this indirectly by verifying the optimizer doesn't crash
        # and returns a price within the anchor day's range
        day0_prices = [4700.0, 4705.0, 4710.0, 4715.0, 4720.0]
        day1_prices = list(np.linspace(4710, 4730, 100))
        day2_prices = [4730.0, 4735.0]
        
        ticks = _make_multiday_ticks([
            (0, day0_prices),
            (1, day1_prices),
            (2, day2_prices),
        ])
        
        best_price, _, _ = self.optimizer.find_optimal_anchor(ticks, 5.0)
        
        self.assertIsNotNone(best_price)
        self.assertGreaterEqual(best_price, 4700.0)
        self.assertLessEqual(best_price, 4720.0)


if __name__ == '__main__':
    unittest.main()
