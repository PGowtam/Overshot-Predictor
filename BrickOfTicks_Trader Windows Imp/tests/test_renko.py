"""
Phase 2 Verification: Renko Builder
Tests per bot_implementation.md Phase 2 Gate.
"""
import unittest
from BrickOfTicks_Trader.data.renko import RenkoBuilder, BrickEvent


class TestRenkoBuilder(unittest.TestCase):

    def test_manual_bricks(self):
        """Gate Test 1: brick_size=10, start=100, feed [100, 105, 110, 115, 95].
        - 100: no brick (no movement from start)
        - 105: no brick (5 < 10)
        - 110: 1 UP brick (open=100, close=110)
        - 115: no brick (5 < 10 for next)
        - 95: NO reversal (needs 2x10=20 from 110, i.e. 90)
        """
        rb = RenkoBuilder(brick_size=10.0, start_price=100.0)

        # 100: no movement
        bricks = rb.update_tick(100.0, 1000)
        self.assertEqual(len(bricks), 0)

        # 105: not enough
        bricks = rb.update_tick(105.0, 2000)
        self.assertEqual(len(bricks), 0)

        # 110: 1 UP brick
        bricks = rb.update_tick(110.0, 3000)
        self.assertEqual(len(bricks), 1)
        self.assertTrue(bricks[0].uptrend)
        self.assertAlmostEqual(bricks[0].open, 100.0)
        self.assertAlmostEqual(bricks[0].close, 110.0)

        # 115: not enough for next (needs 120)
        bricks = rb.update_tick(115.0, 4000)
        self.assertEqual(len(bricks), 0)

        # 95: does NOT trigger reversal (threshold is 110 - 20 = 90)
        bricks = rb.update_tick(95.0, 5000)
        self.assertEqual(len(bricks), 0)

    def test_reversal_at_exact_threshold(self):
        """Verify 90 DOES trigger reversal from up at 110."""
        rb = RenkoBuilder(brick_size=10.0, start_price=100.0)
        rb.update_tick(110.0, 1000)  # 1 UP brick, current=110, uptrend=1

        # 90 = 110 - 2*10 → triggers reversal
        bricks = rb.update_tick(90.0, 2000)
        self.assertEqual(len(bricks), 1)
        self.assertFalse(bricks[0].uptrend)
        # Training code: current -= 2*bs = 110-20=90
        # open = 90+10 = 100, close = 90
        self.assertAlmostEqual(bricks[0].open, 100.0)
        self.assertAlmostEqual(bricks[0].close, 90.0)

    def test_gap_fill_up(self):
        """Gate Test 2: Feed [100, 150] with brick_size=10 → 5 UP bricks."""
        rb = RenkoBuilder(brick_size=10.0, start_price=100.0)
        bricks = rb.update_tick(150.0, 1000)
        self.assertEqual(len(bricks), 5)
        for b in bricks:
            self.assertTrue(b.uptrend)
        # Verify prices: 110, 120, 130, 140, 150
        self.assertAlmostEqual(bricks[0].close, 110.0)
        self.assertAlmostEqual(bricks[1].close, 120.0)
        self.assertAlmostEqual(bricks[2].close, 130.0)
        self.assertAlmostEqual(bricks[3].close, 140.0)
        self.assertAlmostEqual(bricks[4].close, 150.0)

    def test_gap_fill_down(self):
        """Feed [100, 50] with brick_size=10 → 5 DOWN bricks."""
        rb = RenkoBuilder(brick_size=10.0, start_price=100.0)
        bricks = rb.update_tick(50.0, 1000)
        self.assertEqual(len(bricks), 5)
        for b in bricks:
            self.assertFalse(b.uptrend)
        self.assertAlmostEqual(bricks[4].close, 50.0)

    def test_reversal_then_gap(self):
        """After 3 UP, big drop to 70 should give reversal + continuation."""
        rb = RenkoBuilder(brick_size=10.0, start_price=100.0)
        rb.update_tick(130.0, 1000)  # 3 UP bricks, current=130, uptrend=1

        # From 130 uptrend, reversal threshold is 130-20=110
        # Drop to 70: reversal at 110 (current=110), then continuation:
        # 100, 90, 80, 70 (4 more DOWN bricks)
        # Total: 1 reversal + 4 continuation = 5 DOWN bricks? Let's trace:
        # Training code: current_price -= 2*bs = 130-20=110, emit brick(open=110+10=120, close=110)
        # uptrend=-1, then while price<=110-10=100: yes (70<=100)
        #   current=100, emit brick(110,100), continue
        #   while 70<=100-10=90: yes, current=90, emit(100,90)
        #   while 70<=90-10=80: yes, current=80, emit(90,80)
        #   while 70<=80-10=70: yes, current=70, emit(80,70)
        #   while 70<=70-10=60: no, stop
        # Total: 1+4=5 DOWN bricks
        bricks = rb.update_tick(70.0, 2000)
        self.assertEqual(len(bricks), 5)
        self.assertAlmostEqual(bricks[0].close, 110.0)  # Reversal
        self.assertAlmostEqual(bricks[0].open, 120.0)
        self.assertAlmostEqual(bricks[4].close, 70.0)

    def test_sequence_tracking(self):
        """Gate Test 4: 5 UP + 3 DOWN → sequence[-8:] == '11111000'."""
        rb = RenkoBuilder(brick_size=10.0, start_price=100.0)

        # 5 UP bricks
        rb.update_tick(150.0, 1000)
        self.assertEqual(len(rb.history), 5)

        # To get 3 DOWN from current=150, uptrend=1:
        # Reversal at 150-20=130 → current=130, 1 DOWN brick
        # Then continuation: 120, 110 → 2 more DOWN bricks
        # Need price <= 110 after reversal at 130 and continuation at 120
        # So price=110 → reversal at 130 + cont at 120 + cont at 110 = 3 DOWN
        rb.update_tick(110.0, 2000)
        self.assertEqual(len(rb.history), 8)
        self.assertEqual(rb.sequence[-8:], "11111000")

    def test_neutral_state(self):
        """Neutral state uses 1x threshold for both up and down."""
        rb = RenkoBuilder(brick_size=10.0, start_price=100.0)
        self.assertEqual(rb.uptrend, 0)

        # From neutral, going down by 1x should work
        bricks = rb.update_tick(90.0, 1000)
        self.assertEqual(len(bricks), 1)
        self.assertFalse(bricks[0].uptrend)
        self.assertEqual(rb.uptrend, -1)


if __name__ == '__main__':
    unittest.main()
