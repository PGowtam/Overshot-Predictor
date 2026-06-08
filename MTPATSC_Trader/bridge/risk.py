"""
MTPATSC Trader — Risk Manager
===============================
Evaluates runtime safety boundaries including daily drawdown limits,
spread gates, and opportunistic break-even triggers.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Evaluates runtime safety boundaries including daily drawdown limits,
    spread gates, and opportunistic break-even triggers.
    """

    @staticmethod
    def check_daily_limit(daily_pnl_pts: float, brick_size: float) -> bool:
        """
        Returns False if the system has lost the equivalent of 5 stop-losses
        in a single session (-5R).
        """
        limit = -5.0 * brick_size

        if daily_pnl_pts < limit:
            logger.warning(f"DAILY LIMIT EXCEEDED: PnL {daily_pnl_pts:.4f} < Limit {limit:.4f}")
            return False

        return True

    @staticmethod
    def check_position_open(state) -> bool:
        """
        Returns False if there is currently an active position.
        """
        # A ticket of 0 implies no open position
        if state.get('active_ticket', 0) != 0:
            return False
        return True

    @staticmethod
    def check_spread(spread: float, brick_size: float) -> bool:
        """
        Returns False if spread exceeds 10% of brick size.
        This prevents execution during wide-spread conditions.
        """
        if brick_size <= 0:
            return True
        spread_pct = spread / brick_size
        if spread_pct > 0.10:
            logger.warning(f"SPREAD TOO WIDE: {spread:.4f} = {spread_pct*100:.1f}% of brick {brick_size:.4f}")
            return False
        return True

    @staticmethod
    def check_be_trigger(tick: Dict[str, float], state) -> bool:
        """
        Evaluates if the break-even SL modification should be triggered.
        Trigger is 0.3125 * Take Profit distance.
        """
        entry = state.get('active_entry', 0.0)
        tp = state.get('active_tp', 0.0)
        direction = state.get('active_direction', 0)

        if direction == 0 or tp == 0.0 or entry == 0.0:
            return False

        tp_dist = abs(tp - entry)
        trigger_dist = 0.3125 * tp_dist

        if direction == 1:  # BUY
            if tick['bid'] >= entry + trigger_dist:
                return True

        elif direction == -1:  # SELL
            if tick['ask'] <= entry - trigger_dist:
                return True

        return False
