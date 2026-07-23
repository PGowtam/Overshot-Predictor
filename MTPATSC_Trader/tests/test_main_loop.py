import time
import threading
import socket
import pytest
from bridge.main import BridgeEngine

class MockMQL5Client:
    def __init__(self, port=9000):
        self.port = port
        self.sock = None
        
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", self.port))
        
    def send(self, msg: str):
        self.sock.sendall((msg + "\n").encode('utf-8'))
        
    def close(self):
        if self.sock:
            self.sock.close()

@pytest.fixture
def test_engine():
    # Setup test-specific DB with alternate test ports to avoid collisions
    engine = BridgeEngine(tick_port=29000, cmd_port=29001)
    engine.state.filepath = "logs/test_main_state.json"
    engine.state.tmp_filepath = "logs/test_main_state.json.tmp"
    engine.logger.filepath = "logs/test_main_trades.csv"
    
    # Do not block on start() forever. We'll run it in a thread and kill it.
    yield engine

def test_dry_run_warmup(test_engine):
    """
    Simulates MQL5 sending DAYOPEN and HDONE, verifying the BridgeEngine
    correctly transitions through warmup.
    """
    engine = test_engine
    
    # Mock the ensemble load so it doesn't fail if TF is missing or models are slow
    engine.ensemble.load = lambda: None
    
    # Run the engine startup in a background thread so we can send mock socket messages
    def run_engine():
        try:
            engine.start()
        except SystemExit:
            pass # Expected when we close it
            
    thread = threading.Thread(target=run_engine, daemon=True)
    thread.start()
    
    # Give the server a moment to bind
    time.sleep(0.5)
    
    client = MockMQL5Client(port=29000)
    try:
        client.connect()
        
        # 1. Send DAYOPEN
        client.send("DAYOPEN|123456789|2400.00")
        
        # 2. Send multi-day history ticks for PathOptimizer
        # PathOptimizer needs >=2 UTC day boundaries.
        # Day 1 (anchor): timestamps at day_num = X
        # Day 2 (target): timestamps at day_num = X+1
        # K=0.00295 * 2400 = 7.08 brick size
        # We need ~200 points of upward movement to form ~28 bricks
        price = 2400.00
        
        # Day 1: base_ts such that ts//1000//86400 = some day
        day1_base = 86400 * 20000 * 1000  # Some UTC day in ms
        for i in range(600):
            price += 0.5  # Gradual movement for anchor day
            ts = day1_base + i * 60000  # 1 min apart
            client.send(f"HTICK|{ts}|{price:.2f}|{price+0.1:.2f}|1.0|1.0")
        
        # Day 2: next UTC day (target day)
        day2_base = day1_base + 86400000  # +1 day in ms
        for i in range(600):
            price += 0.5  # Continue trending up
            ts = day2_base + i * 60000
            client.send(f"HTICK|{ts}|{price:.2f}|{price+0.1:.2f}|1.0|1.0")
        
        # Let socket buffer fully drain before sending HDONE
        time.sleep(1.0)
            
        # 3. Send HDONE
        client.send("HDONE|1200")
        
        # Poll for warmup completion (up to 15 seconds)
        # This handles Numba JIT compilation variability
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if engine.renko.brick_count > 0:
                break
            time.sleep(0.5)
        
        # PathOptimizer should have found an anchor and replayed bricks
        assert engine.renko.brick_count > 10, f"Expected >10 bricks, got {engine.renko.brick_count}"
        # With 1200 ticks all fed to feature engine, z_ofi > 1000 → warmup passed
        assert engine.state.get("warmup_done") is True
        
    finally:
        client.close()
        
def test_latency_profiling(test_engine):
    """
    Phase 7.4 Latency Profiling
    """
    engine = test_engine
    engine.renko.update_brick_size(2400.00)
    
    tick = {'bid': 2400.00, 'ask': 2400.10, 'bid_vol': 1.0, 'ask_vol': 1.0, 'time_msc': 10000}
    
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        # _process_tick computes features and updates buffer/renko
        engine._process_tick(tick, is_warmup=True)
        latencies.append((time.perf_counter() - t0) * 1000)
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average tick latency: {avg_latency:.3f}ms")
    
    # Assert _process_tick average < 10ms
    assert avg_latency < 10.0
