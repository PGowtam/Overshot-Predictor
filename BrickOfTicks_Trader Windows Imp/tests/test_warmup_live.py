"""
Phase 7.2: Warmup Live Verification

Initializes the OrbitEngine, runs a 10,000 tick warmup, 
and verifies the internal state for readiness.
"""
import time
import MetaTrader5 as mt5
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from BrickOfTicks_Trader.main import OrbitEngine
from BrickOfTicks_Trader.config.settings import SYMBOL, WARMUP_TICKS

def verify_warmup():
    print(f"--- STARTING WARMUP VERIFICATION (7.2) ---")
    engine = OrbitEngine()
    
    # 1. Initialize (Connects to MT5, Loads Models)
    if not engine.initialize():
        print("FAILED: Initialization error.")
        return

    # 2. Execute Warmup (Large count)
    start_time = time.time()
    engine._warmup(WARMUP_TICKS)
    elapsed = time.time() - start_time
    
    # 3. VERIFICATIONS
    print(f"Warmup processed {len(engine.buffer.snapshots)} bricks.")
    
    # 3. VERIFICATIONS
    
    # A. Timing Check (< 30s)
    if elapsed < 30.0:
        print(f"SUCCESS: Warmup completed in {elapsed:.2f}s (< 30s).")
    else:
        print(f"FAILED: Warmup took too long: {elapsed:.2f}s.")

    # B. Z-Score Saturation (Check if window is full)
    z_len = len(engine.features.z_ofi.deque)
    if z_len >= 1000:
        print(f"SUCCESS: Z-Score windows saturated (Length: {z_len}).")
    else:
        print(f"WARNING: Z-Score window only partially filled (Length: {z_len}).")

    # C. Renko Price Sync
    tick = mt5.symbol_info_tick(SYMBOL)
    diff = abs(engine.renko.current_price - tick.bid)
    if diff < (engine.renko.brick_size * 2): # Allow some drift depending on volatility
        print(f"SUCCESS: Renko price synced. Internal: {engine.renko.current_price:.2f}, Market: {tick.bid:.2f} (Diff: {diff:.4f})")
    else:
        print(f"FAILED: Renko price out of sync. Diff: {diff:.4f}")

    # D. Buffer Snapshots (Need >= 10 for first inference)
    snap_count = len(engine.buffer.snapshots)
    if snap_count >= 10:
        print(f"SUCCESS: Inference Buffer ready (Snapshots: {snap_count}).")
    else:
        print(f"FAILED: Not enough bricks formed during warmup (Snapshots: {snap_count}). Need 10.")

    # E. No-Trade Check
    # We haven't placed any trades, but we'll check terminal 
    # to ensure no magic number orders were sent.
    positions = mt5.positions_get(symbol=SYMBOL)
    ours = [p for p in positions if p.magic == 314159] if positions else []
    if len(ours) == 0:
        print("SUCCESS: Zero trades placed during warmup.")
    else:
        print(f"WARNING: Found {len(ours)} active trades during warmup.")

    print(f"--- WARMUP VERIFICATION FINISHED ---")
    mt5.shutdown()

if __name__ == "__main__":
    verify_warmup()
