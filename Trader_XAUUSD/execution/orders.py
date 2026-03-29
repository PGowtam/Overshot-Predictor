import MetaTrader5 as mt5
import time
from config.settings import SYMBOL, LOT_SIZE, DEVIATION, MAGIC_NUMBER
from utils.logger import logger

class OrderExecutor:
    def __init__(self):
        pass
        
    def send_market_order(self, action, sl_price=None, tp_price=None):
        """
        Action: 1 (BUY), -1 (SELL)
        """
        # Determine Type
        order_type = mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).ask if action == 1 else mt5.symbol_info_tick(SYMBOL).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(LOT_SIZE),
            "type": order_type,
            "price": price,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "RL_Ensemble",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if sl_price:
            request["sl"] = float(sl_price)
        if tp_price:
            request["tp"] = float(tp_price)
            
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order Send Failed: {result.retcode} - {result.comment}")
            return None
            
        logger.info(f"Order Executed: {result.order} | Price: {result.price} | SL: {sl_price}")
        return result.order

    def modify_sl(self, ticket, new_sl):
        # Get existing position to keep TP
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Ticket {ticket} not found for modification.")
            return False
            
        pos = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": SYMBOL,
            "position": ticket, # Position ID
            "sl": float(new_sl),
            "tp": pos.tp,
            "magic": MAGIC_NUMBER,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"SL Modify Failed: {result.retcode}")
            return False
            
        logger.info(f"Modified SL for {ticket} to {new_sl}")
        return True

    def move_sl_to_entry(self, symbol):
        """
        Moves SL to Break-Even (Position Open Price).
        Checks if already moved to avoid spam.
        """
        positions = mt5.positions_get(symbol=symbol, magic=MAGIC_NUMBER)
        if not positions:
            return False
            
        pos = positions[0] # Assume one position per symbol logic
        entry_price = pos.price_open
        current_sl = pos.sl
        
        # Check direction
        is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
        
        needs_update = False
        if is_buy:
            # If SL is below entry, move it up.
            # If SL is already >= entry, ignore.
            if current_sl < entry_price - 0.00001: # Epsilon
                needs_update = True
        else:
            # Sell: If SL is above entry, move it down.
            # If SL <= entry, ignore.
            if current_sl > entry_price + 0.00001 or current_sl == 0.0:
                needs_update = True
                
        if needs_update:
            logger.info(f"Triggering BE for Ticket {pos.ticket}. Entry: {entry_price}")
            return self.modify_sl(pos.ticket, entry_price)
            
        return False
        
    def close_all(self):
        positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
        if positions:
            for pos in positions:
                type_close = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(SYMBOL).bid if type_close == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(SYMBOL).ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "position": pos.ticket,
                    "volume": pos.volume,
                    "type": type_close,
                    "price": price,
                    "deviation": DEVIATION,
                    "magic": MAGIC_NUMBER,
                }
                mt5.order_send(request)
                logger.info(f"Closed position {pos.ticket}")

    def send_limit_order(self, action, price, sl, tp):
        """
        Sends a Limit Order at a specific price.
        """
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if action == 1 else mt5.ORDER_TYPE_SELL_LIMIT
        
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": SYMBOL,
            "volume": float(LOT_SIZE),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "RL_Limit_Fallback",
            "type_time": mt5.ORDER_TIME_DAY, # Expires at end of day
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Limit Order Failed: {result.retcode} - {result.comment}")
            return None
            
        logger.info(f"Limit Order Placed: {result.order} @ {price}")
        return result.order

    def cancel_order(self, ticket):
        """
        Cancels a pending order.
        """
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
            "magic": MAGIC_NUMBER,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Cancel Failed for {ticket}: {result.retcode}")
            return False
            
        logger.info(f"Cancelled Order {ticket}")
        return True
