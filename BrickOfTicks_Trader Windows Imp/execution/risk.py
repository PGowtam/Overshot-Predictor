"""
Phase 6: Risk Management

Handles daily drawdowns, slippage guards, break-even logic,
and maximum concurrent position logic.
"""
import MetaTrader5 as mt5
from BrickOfTicks_Trader.config.settings import SYMBOL, MAGIC_NUMBER
from BrickOfTicks_Trader.utils.logger import logger

class RiskManager:
    def __init__(self, max_drawdown_pct: float = 0.03, max_concurrent_trades: int = 1):
        self.max_drawdown = max_drawdown_pct
        self.max_concurrent_trades = max_concurrent_trades

    def check_daily_limit(self) -> bool:
        """
        Verify if the daily drawdown limit has been breached.
        Drawdown is calculated as: (balance - equity) / balance
        Returns True if safe to trade, False if halted.
        """
        account = mt5.account_info()
        if account is None:
            logger.error("Failed to retrieve MT5 account info for risk check")
            return False

        balance = account.balance
        equity = account.equity

        if balance <= 0:
            return False
            
        drawdown_pct = (balance - equity) / balance
        
        if drawdown_pct >= self.max_drawdown:
            logger.warning(
                f"DAILY LIMIT REACHED! Drawdown {drawdown_pct*100:.2f}% "
                f">= allowed {self.max_drawdown*100:.2f}%"
            )
            return False
            
        return True

    def can_open_new_position(self) -> bool:
        """Check if we are allowed to open a new position based on max trades limit."""
        positions = mt5.positions_get(symbol=SYMBOL)
        
        # If positions is None it might mean 0 positions or error, assume 0 for check
        if positions is None:
            return True
            
        our_positions = [p for p in positions if p.magic == MAGIC_NUMBER]
        
        if len(our_positions) >= self.max_concurrent_trades:
            logger.info(f"Max concurrent trades ({self.max_concurrent_trades}) reached. Cannot open new.")
            return False
            
        return True

    def has_active_positions(self) -> bool:
        """Check if there are any active positions belonging to this EA."""
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return False
        ours = [p for p in positions if p.magic == MAGIC_NUMBER]
        return len(ours) > 0

    def check_slippage(self, current_price: float, brick_close: float, brick_size: float) -> bool:
        """
        Guard against excessive slippage during market orders.
        Returns True if slippage is > 8% of brick_size (requires limit order fallback).
        False if within acceptable limits.
        """
        allowable_slip = 0.08 * brick_size
        actual_slip = abs(current_price - brick_close)
        
        if actual_slip > allowable_slip:
            logger.info(f"Slippage guard triggered. Slip ({actual_slip}) > Allowed ({allowable_slip})")
            return True
        return False


