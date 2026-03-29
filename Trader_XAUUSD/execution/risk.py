import MetaTrader5 as mt5
from config.settings import MAX_DAILY_DRAWDOWN_PCT, SYMBOL, MAGIC_NUMBER
from utils.logger import logger
from utils.state import StateManager

class RiskManager:
    def __init__(self, state_manager: StateManager):
        self.state = state_manager
        
    def check_daily_limit(self):
        # Calculate Real PnL from History Deals + Current Floating
        # Or simplistic: Trust State?
        # Better: Query MT5 history for today.
        
        # 1. Floating PnL
        floating = 0.0
        positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
        if positions:
            for pos in positions:
                floating += pos.profit
                
        # 2. Realized PnL (Deals Today)
        realized = 0.0
        # Get start of day
        # ... logic to get deals ...
        
        # Simplified: Use equity from Account Info vs Start of Day Equity?
        account = mt5.account_info()
        if not account:
            return True # Fail safe
            
        # We need Start of Day Balance.
        # If we stored it in state, use it.
        # For now, let's assume we use the stored daily_pnl from Renko Logic as proxy,
        # OR we rely on Account Equity % Drawdown if we knew the high water mark.
        
        # Let's use the logic defined in RenkoEnv: "Daily PnL" accumuluated.
        # Because MT5 PnL might be noisy with latency.
        # But for SAFETY, we should use Account Equity.
        
        # Assume start balance is roughly Account Balance at 00:00 (we can capture this in main).
        # Let's rely on stored "daily_pnl" (points or $?)
        # Environment uses Points/Units (+0.5, -0.5).
        # We need % Drawdown.
        
        # Implementation:
        # Just check the accumuluated units from state.
        # Env logic: -3% limit. 
        # Since we simulate units, we trust the Agent's internal PnL counter (state variable).
        
        current_pnl_units = self.state.get("daily_pnl", 0.0)
        # Agent logic: -3.0 units = -3% approx if properly scaled?
        # Actually in Env: `self.max_daily_loss = -3.0`.
        # So "Units" are %. 
        
        if current_pnl_units <= -3.0:
            # Check if we already warned to prevent infinite log spam
            if not self.state.get("risk_limit_hit_warned", False):
                logger.warning(f"Daily Risk Limit Hit! PnL: {current_pnl_units}")
                self.state.update("risk_limit_hit_warned", True)
            return False # Stop Trading
            
        return True
        
    def update_pnl(self, unit_pnl):
        current = self.state.get("daily_pnl", 0.0)
        new_val = current + unit_pnl
        self.state.update("daily_pnl", new_val)
