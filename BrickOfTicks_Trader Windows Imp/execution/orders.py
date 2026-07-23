"""
Phase 6: Order Execution

Places, modifies, and closes orders via MT5.
Supports market orders, limit orders for slippage guards, and SL modifications.
"""
import MetaTrader5 as mt5
from BrickOfTicks_Trader.config.settings import SYMBOL, LOT_SIZE, MAGIC_NUMBER, DEVIATION, FILLING_MODE
from BrickOfTicks_Trader.utils.logger import logger

class OrderExecutor:
    def __init__(self):
        # Determine filling mode flag based on config
        self.filling_mode_flag = mt5.ORDER_FILLING_IOC
        if FILLING_MODE.upper() == "FOK":
            self.filling_mode_flag = mt5.ORDER_FILLING_FOK
        elif FILLING_MODE.upper() == "RETURN":
            self.filling_mode_flag = mt5.ORDER_FILLING_RETURN
            
        # Cache symbol digits for normalization
        info = mt5.symbol_info(SYMBOL)
        self.digits = info.digits if info else 2

    def _normalize(self, price: float) -> float:
        """Round price to the correct number of digits for the symbol."""
        return round(float(price), self.digits)

    def send_market_order(self, direction: int, sl: float, tp: float, comment: str = "") -> int:
        """
        Place a market order.
        direction: 1 (BUY) or -1 (SELL)
        Returns ticket number or None on failure.
        """
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            logger.error(f"Failed to get tick for {SYMBOL} before placing market order")
            return None
            
        price = tick.ask if direction == 1 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(LOT_SIZE),
            "type": order_type,
            "price": self._normalize(price),
            "sl": self._normalize(sl),
            "tp": self._normalize(tp),
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.filling_mode_flag,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"mt5.order_send failed for MARKET command, no response")
            return None
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Market order failed: retcode={result.retcode}, action={request['action']}")
            logger.error(f"Request dict: {request}")
            return None
            
        logger.info(f"Market order placed successfully. Ticket: {result.order}")
        return result.order

    def send_limit_order(self, direction: int, price: float, sl: float, tp: float, comment: str = "") -> int:
        """
        Place a limit order (used as fallback for slippage).
        direction: 1 (BUY_LIMIT) or -1 (SELL_LIMIT)
        Returns ticket number or None on failure.
        """
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == 1 else mt5.ORDER_TYPE_SELL_LIMIT
        
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": SYMBOL,
            "volume": float(LOT_SIZE),
            "type": order_type,
            "price": self._normalize(price),
            "sl": self._normalize(sl),
            "tp": self._normalize(tp),
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.filling_mode_flag,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Limit order failed: result={result if result else 'None'}")
            return None
            
        logger.info(f"Limit order placed successfully. Ticket: {result.order}")
        return result.order

    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify the StopLoss of an existing position."""
        position = self.get_position(ticket)
        if not position:
            logger.error(f"Cannot modify SL for ticket {ticket}: position not found")
            return False
            
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": SYMBOL,
            "sl": self._normalize(new_sl),
            "tp": self._normalize(position.tp)  # Keep existing TP
        }
        
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"SL modification failed: result={result if result else 'None'}")
            return False
            
        logger.info(f"SL modified successfully for ticket {ticket} to {new_sl}")
        return True

    def cancel_pending(self, ticket: int) -> bool:
        """Cancel an unfilled pending limit order."""
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket
        }
        
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Pending order cancellation failed: result={result if result else 'None'}")
            return False
            
        logger.info(f"Pending order {ticket} cancelled successfully")
        return True

    def close_position(self, ticket: int) -> bool:
        """Closes an open position using an opposite market order."""
        position = self.get_position(ticket)
        if not position:
            logger.error(f"Cannot close position {ticket}: not found")
            return False

        # Opposite direction
        opp_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            return False
            
        price = tick.bid if opp_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": position.volume,
            "type": opp_type,
            "position": ticket,
            "price": price,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.filling_mode_flag,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close position failed: result={result if result else 'None'}")
            return False
            
        logger.info(f"Position {ticket} closed successfully")
        return True

    def get_position(self, ticket: int):
        """Retrieve a specific position by ticket."""
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            return None
        return positions[0]

    def has_active_positions(self) -> bool:
        """Check if there are any active positions for our EA."""
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None:
            return False
        
        # Filter by our magic number
        our_positions = [p for p in positions if p.magic == MAGIC_NUMBER]
        return len(our_positions) > 0
