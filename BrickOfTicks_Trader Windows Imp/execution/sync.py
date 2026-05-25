"""
Phase 11: Daily Synchronization
Handles the UTC midnight rollover and fetches the new D1 open price 
to update the internal brick sizing dynamically.
"""
import time
from datetime import datetime
import MetaTrader5 as mt5

from BrickOfTicks_Trader.config.settings import SYMBOL, BRICK_SIZE_FACTOR
from BrickOfTicks_Trader.utils.logger import logger


class DailySynchronizer:
    def __init__(self, current_day: int = None):
        """
        Args:
            current_day (int): The day of the year (0-365) the bot thinks it is currently.
                               If None, initializes on the first check.
        """
        self.current_day = current_day

    def check_and_sync(self) -> float:
        """
        Checks if a new UTC day has started. 
        If true, fetches the MT5 D1 open price and returns the new brick size.
        If false (same day), returns None.
        """
        # Get current UTC time from system (trading servers generally follow UTC or standard broker offsets)
        now_dt = datetime.utcnow()
        today = now_dt.timetuple().tm_yday

        if self.current_day is None:
            self.current_day = today
            # At startup, we don't return a new brick size here, 
            # as the initial setup handles it via current Ask price.
            return None

        if today != self.current_day:
            logger.info(f"Daily Rollover detected! UTC Day: {self.current_day} -> {today}")
            self.current_day = today
            return self._fetch_new_brick_size()
            
        return None

    def _fetch_new_brick_size(self) -> float:
        """
        Fetches the open price of the current D1 candle and multiplies by the factor.
        """
        # copy_rates_from_pos(symbol, timeframe, start_pos, count)
        # Position 0 is the current unfinished daily candle
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 1)
        
        if rates is None or len(rates) == 0:
            logger.error("Failed to fetch D1 open price during daily sync.")
            return None
            
        open_price = float(rates[0]['open'])
        new_brick_size = open_price * BRICK_SIZE_FACTOR
        
        logger.info(f"Daily Sync: D1 Open = {open_price:.2f} | New Brick Size = {new_brick_size:.4f}")
        return new_brick_size
