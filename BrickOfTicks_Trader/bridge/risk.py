import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Evaluates runtime safety boundaries including daily drawdown limits 
    and opportunistic break-even triggers.
    """
    
    @staticmethod
    def check_daily_limit(daily_pnl_pts: float, brick_size: float) -> bool:
        """
        Returns False if the system has lost the equivalent of 5 stop-losses
        in a single session.
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
    def check_be_trigger(tick: Dict[str, float], state) -> bool:
        """
        Evaluates if the break-even SL modification should be triggered.
        # 0.3125 = 5/16ths of a brick — chosen to balance premature exits vs protection
        """
        entry = state.get('active_entry', 0.0)
        bs = state.get('active_brick_size', 0.0)
        direction = state.get('active_direction', 0)
        
        if direction == 0 or bs <= 0:
            return False
            
        trigger_dist = 0.3125 * bs
        
        if direction == 1: # BUY
            if tick['bid'] >= entry + trigger_dist:
                return True
                
        elif direction == -1: # SELL
            if tick['ask'] <= entry - trigger_dist:
                return True
                
        return False
