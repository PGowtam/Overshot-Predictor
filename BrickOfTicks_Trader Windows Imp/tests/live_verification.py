"""
Phase 6.4: Live Integration Verification

This script performs a real-world test on the MT5 Demo Terminal.
- Connection
- Symbol Selection
- 0.01 lot Order Placement (XAUUSD)
- SL Modification
- Close Position
- Filling Mode confirmation
"""
import MetaTrader5 as mt5
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from BrickOfTicks_Trader.config.settings import SYMBOL, LOT_SIZE, MAGIC_NUMBER
from BrickOfTicks_Trader.execution.orders import OrderExecutor
from BrickOfTicks_Trader.execution.risk import RiskManager
from BrickOfTicks_Trader.utils.logger import logger

def run_live_test():
    logger.info("--- STARTING LIVE INTEGRATION TEST (6.4) ---")
    
    if not mt5.initialize():
        print(f"FAILED: MT5 Initialization Error: {mt5.last_error()}")
        return

    # 1. Symbol Check
    if not mt5.symbol_select(SYMBOL, True):
        print(f"FAILED: Symbol {SYMBOL} not found or selectable.")
        mt5.shutdown()
        return
    
    # Wait for price data to populate
    import time
    time.sleep(2)
    
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None or tick.ask <= 0:
        print(f"FAILED: No price data for {SYMBOL} yet. Try later.")
        mt5.shutdown()
        return

    symbol_info = mt5.symbol_info(SYMBOL)
    print(f"Connected to Symbol: {SYMBOL}")
    print(f"Digits: {symbol_info.digits}, Spread: {tick.ask - tick.bid:.2f}")

    executor = OrderExecutor()
    risk = RiskManager()

    # 2. Daily Risk Check
    if not risk.check_daily_limit():
        print("FAILED: Risk Manager blocked execution (Daily limit breached or account info error).")
        mt5.shutdown()
        return
    print("Risk Manager: Daily limit check passed.")

    # 3. Placement Test (BUY 0.01)
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask
    sl = price - 5.0 # 5 points away
    tp = price + 5.0
    
    print(f"Attempting Market BUY 0.01 at {price}...")
    ticket = executor.send_market_order(direction=1, sl=sl, tp=tp, comment="Live_Verify_6.4")
    
    if ticket:
        print(f"SUCCESS: Market Order Placed. Ticket: {ticket}")
    else:
        print(f"FAILED: Market Order placement failed. Check logs/terminal.")
        mt5.shutdown()
        return

    # 4. Modification Test (Move SL up by 1 point)
    new_sl = sl + 1.0
    print(f"Attempting SL Modification for ticket {ticket} to {new_sl}...")
    if executor.modify_sl(ticket, new_sl):
        print(f"SUCCESS: SL Modified.")
    else:
        print(f"FAILED: SL Modification failed.")

    # 5. Position Check
    pos = executor.get_position(ticket)
    if pos:
        print(f"SUCCESS: Position verified in list style. Current Profit: {pos.profit}")
    else:
        print(f"FAILED: Position {ticket} not found in MT5 after placement.")

    # 6. Close Test
    print(f"Attempting to Close Position {ticket}...")
    if executor.close_position(ticket):
        print("SUCCESS: Position closed.")
    else:
        print("FAILED: Position closing failed.")

    # 7. Filling Type Log
    terminal = mt5.terminal_info()
    print(f"Broker/Terminal Filling Mode: {executor.filling_mode_flag}")
    
    print("--- LIVE INTEGRATION TEST FINISHED ---")
    mt5.shutdown()

if __name__ == "__main__":
    run_live_test()
