"""
Phase 6: Order Execution Tests

Tests order execution logic by mocking MetaTrader5 API actions.
"""
import unittest
from unittest.mock import patch, MagicMock
from BrickOfTicks_Trader.execution.orders import OrderExecutor

class TestOrderExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = OrderExecutor()
        
    @patch('BrickOfTicks_Trader.execution.orders.mt5')
    def test_send_market_order_buy_success(self, mock_mt5):
        """Verify market BUY sets correct dictionaries and returns ticket."""
        mock_mt5.ORDER_FILLING_IOC = 1
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.TRADE_RETCODE_DONE = 10009
        
        # Mock tick
        mock_tick = MagicMock()
        mock_tick.ask = 2100.5
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        # Mock successful order send
        mock_result = MagicMock()
        mock_result.retcode = 10009
        mock_result.order = 12345
        mock_mt5.order_send.return_value = mock_result
        
        ticket = self.executor.send_market_order(direction=1, sl=2090.0, tp=2110.0)
        
        self.assertEqual(ticket, 12345)
        mock_mt5.order_send.assert_called_once()
        
        # Verify the request dict
        request = mock_mt5.order_send.call_args[0][0]
        self.assertEqual(request['type'], 0) # BUY
        self.assertEqual(request['price'], 2100.5) # Ask used for BUY
        self.assertEqual(request['sl'], 2090.0)
        self.assertEqual(request['tp'], 2110.0)

    @patch('BrickOfTicks_Trader.execution.orders.mt5')
    def test_modify_sl(self, mock_mt5):
        """Verify modifying stop loss."""
        mock_mt5.TRADE_ACTION_SLTP = 6
        mock_mt5.TRADE_RETCODE_DONE = 10009
        
        mock_pos = MagicMock()
        mock_pos.tp = 2110.0
        mock_mt5.positions_get.return_value = [mock_pos]
        
        mock_res = MagicMock()
        mock_res.retcode = 10009
        mock_mt5.order_send.return_value = mock_res
        
        success = self.executor.modify_sl(12345, 2095.0)
        
        self.assertTrue(success)
        mock_mt5.order_send.assert_called_once()
        request = mock_mt5.order_send.call_args[0][0]
        self.assertEqual(request['action'], 6) # SLTP
        self.assertEqual(request['sl'], 2095.0)
        self.assertEqual(request['tp'], 2110.0)

    @patch('BrickOfTicks_Trader.execution.orders.mt5')
    def test_close_position(self, mock_mt5):
        """Verify closing a BUY position results in a SELL market order."""
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.TRADE_RETCODE_DONE = 10009
        
        mock_pos = MagicMock()
        mock_pos.type = 0 # BUY
        mock_pos.volume = 0.01
        mock_mt5.positions_get.return_value = [mock_pos]
        
        mock_tick = MagicMock()
        mock_tick.bid = 2101.0
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        mock_res = MagicMock()
        mock_res.retcode = 10009
        mock_mt5.order_send.return_value = mock_res
        
        success = self.executor.close_position(12345)
        self.assertTrue(success)
        
        req = mock_mt5.order_send.call_args[0][0]
        self.assertEqual(req['type'], 1) # Must be SELL to close BUY
        self.assertEqual(req['position'], 12345)
        self.assertEqual(req['price'], 2101.0) # Bid used for SELL

if __name__ == '__main__':
    unittest.main()
