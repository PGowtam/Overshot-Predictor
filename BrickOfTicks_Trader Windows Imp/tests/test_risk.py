"""
Phase 6 Verification: Risk Management Tests
"""
import unittest
from unittest.mock import patch, MagicMock
from BrickOfTicks_Trader.execution.risk import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.risk = RiskManager(max_drawdown_pct=0.03, max_concurrent_trades=1)
        
    @patch('BrickOfTicks_Trader.execution.risk.mt5')
    def test_daily_limit_safe(self, mock_mt5):
        """Verify check_daily_limit allows trading when DD < 3%."""
        mock_acc = MagicMock()
        mock_acc.balance = 10000.0
        mock_acc.equity = 9800.0 # 2% drawdown
        mock_mt5.account_info.return_value = mock_acc
        
        self.assertTrue(self.risk.check_daily_limit())

    @patch('BrickOfTicks_Trader.execution.risk.mt5')
    def test_daily_limit_breached(self, mock_mt5):
        """Verify check_daily_limit halts trading when DD >= 3%."""
        mock_acc = MagicMock()
        mock_acc.balance = 10000.0
        mock_acc.equity = 9600.0 # 4% drawdown
        mock_mt5.account_info.return_value = mock_acc
        
        self.assertFalse(self.risk.check_daily_limit())

    @patch('BrickOfTicks_Trader.execution.risk.mt5')
    def test_max_concurrent_trades(self, mock_mt5):
        """Verify limitation on concurrent positions."""
        from BrickOfTicks_Trader.config.settings import MAGIC_NUMBER
        
        # Scenario: 0 positions
        mock_mt5.positions_get.return_value = []
        self.assertTrue(self.risk.can_open_new_position())
        
        # Scenario: 1 position already open
        mock_pos = MagicMock()
        mock_pos.magic = MAGIC_NUMBER
        mock_mt5.positions_get.return_value = [mock_pos]
        self.assertFalse(self.risk.can_open_new_position())
        
        # Scenario: 1 position but different EA (magic mismatch)
        mock_pos_other = MagicMock()
        mock_pos_other.magic = 999999
        mock_mt5.positions_get.return_value = [mock_pos_other]
        self.assertTrue(self.risk.can_open_new_position())

    def test_slippage_guard(self):
        """Guard against >8% brick size slippage."""
        brick_size = 10.0
        brick_close = 2000.0
        
        # Allowable is 0.8
        self.assertFalse(self.risk.check_slippage(2000.5, brick_close, brick_size)) # 0.5 slip (safe)
        self.assertTrue(self.risk.check_slippage(2000.9, brick_close, brick_size))  # 0.9 slip (halt)
        self.assertTrue(self.risk.check_slippage(1999.1, brick_close, brick_size))  # 0.9 slip down (halt)

    def test_break_even_logic(self):
        """Verify break even trigger and condition."""
        entry = 2000.0
        brick_size = 10.0
        # Req move = 3.125
        
        # BUY
        trigger = self.risk.get_break_even_trigger(entry, 1, brick_size)
        self.assertEqual(trigger, 2003.125)
        self.assertFalse(self.risk.should_move_to_be(2002.0, entry, 1, brick_size))
        self.assertTrue(self.risk.should_move_to_be(2004.0, entry, 1, brick_size))
        
        # SELL
        trigger = self.risk.get_break_even_trigger(entry, -1, brick_size)
        self.assertEqual(trigger, 1996.875)
        self.assertFalse(self.risk.should_move_to_be(1998.0, entry, -1, brick_size))
        self.assertTrue(self.risk.should_move_to_be(1995.0, entry, -1, brick_size))

if __name__ == '__main__':
    unittest.main()
