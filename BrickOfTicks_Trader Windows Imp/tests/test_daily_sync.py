import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from BrickOfTicks_Trader.execution.sync import DailySynchronizer
from BrickOfTicks_Trader.config.settings import BRICK_SIZE_FACTOR

class TestDailySync(unittest.TestCase):
    
    @patch('BrickOfTicks_Trader.execution.sync.datetime')
    def test_initialization(self, mock_dt):
        """On first run, it should just set current_day and return None."""
        mock_now = datetime(2026, 4, 1, 15, 0)
        mock_dt.utcnow.return_value = mock_now
        
        sync = DailySynchronizer()
        res = sync.check_and_sync()
        
        self.assertIsNone(res)
        self.assertEqual(sync.current_day, mock_now.timetuple().tm_yday)
        
    @patch('BrickOfTicks_Trader.execution.sync.mt5')
    @patch('BrickOfTicks_Trader.execution.sync.datetime')
    def test_rollover_triggers_fetch(self, mock_dt, mock_mt5):
        """When UTC day changes, it should return new brick size."""
        # 1. Start on Day 90
        mock_now_1 = datetime(2026, 3, 31, 23, 59)
        mock_dt.utcnow.return_value = mock_now_1
        sync = DailySynchronizer()
        sync.check_and_sync()
        self.assertEqual(sync.current_day, 90)
        
        # 2. Roll to Day 91
        mock_now_2 = datetime(2026, 4, 1, 0, 1)
        mock_dt.utcnow.return_value = mock_now_2
        
        # Mock mt5 return for copy_rates_from_pos
        mock_mt5.copy_rates_from_pos.return_value = [{'open': 2000.0}]
        mock_mt5.TIMEFRAME_D1 = 16408
        
        new_size = sync.check_and_sync()
        self.assertIsNotNone(new_size)
        self.assertAlmostEqual(new_size, 2000.0 * BRICK_SIZE_FACTOR)
        self.assertEqual(sync.current_day, 91)
        
if __name__ == '__main__':
    unittest.main()
