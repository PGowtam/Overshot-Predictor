import os
import json
import time
from bridge.main import BridgeEngine

def test_crash_recovery_simulation():
    print("\n--- Simulating 8.4 Crash Recovery ---")
    
    # 1. Start Engine A and simulate an open trade
    engine_a = BridgeEngine()
    engine_a.state.update("active_ticket", 99999)
    engine_a.state.update("active_dir", 1)
    engine_a.state.update("active_entry", 2400.00)
    engine_a.state.update("active_sl", 2390.00)
    engine_a.state.update("active_tp", 2410.00)
    
    # Assert state file is written safely
    engine_a.state.save()
    assert os.path.exists(engine_a.state.filepath)
    
    print("Engine A state saved. Simulating hard crash (kill -9).")
    
    # Destroy Engine A (simulated crash)
    del engine_a
    
    # 2. Boot Engine B (Simulating restart)
    print("Restarting... Booting Engine B.")
    engine_b = BridgeEngine()
    
    # Wait for state file read
    ticket = engine_b.state.get("active_ticket")
    assert ticket == 99999, f"State did not persist active ticket. Got {ticket}"
    assert engine_b.state.get("active_sl") == 2390.00
    print("Engine B successfully reloaded state.json. Active ticket recognized.")
    
    # 3. Simulate another BUY signal. It should NOT send a command.
    print("Triggering new BUY signal to test duplicate prevention.")
    
    # We will mock sender to ensure it's not called
    sent_command = False
    def mock_buy(*args, **kwargs):
        nonlocal sent_command
        sent_command = True
        return {'status': 'OK', 'ticket': 88888}
        
    engine_b.sender.buy = mock_buy
    
    class FakeBrick:
        close = 2405.00
        uptrend = 1
        timestamp = int(time.time() * 1000)
        brick_size = 7.08
        sequence = "1"
        open = 2397.92
        high = 2405.00
        low = 2397.92
    
    fake_tensors = (None, None)
    
    # Mock ensemble to return a BUY signal
    engine_b.ensemble.predict = lambda x, y: {'action': 1, 'votes': 3}
    
    # Force the signal
    engine_b._on_signal(FakeBrick(), fake_tensors)
    
    # Assert no duplicate order
    assert sent_command == False, "Engine B fired a duplicate trade while one was already active!"
    print("Duplicate prevention ASSERT PASSED. No duplicate trades opened.")
    
    # Ensure existing SL/TP is untouched
    assert engine_b.state.get("active_sl") == 2390.00
    
    print("Crash Recovery Simulation PASSED! Phase 8.4 is ready.")
    
if __name__ == "__main__":
    test_crash_recovery_simulation()
