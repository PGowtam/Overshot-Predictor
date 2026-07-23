import pytest
from bridge.feature_engine import LiveFeatureEngine, RollingZScore

def test_first_tick_returns_zeros():
    engine = LiveFeatureEngine()
    vec = engine.compute_vector(2400.00, 2400.10, 5.0, 3.0, 1000)
    assert vec == [0.0] * 9

def test_volume_fallback():
    engine = LiveFeatureEngine()
    # Init tick
    engine.compute_vector(2400.00, 2400.10, 0.0, 0.0, 1000)
    
    # Tick with higher mid, but 0 volume
    # Mid 1 = 2400.05
    # Mid 2 = 2400.25 -> UP
    vec = engine.compute_vector(2400.20, 2400.30, 0.0, 0.0, 1001)
    
    # The raw_ofi inside should be 1.0, depth 0.0, susc 0.0
    # Because z-score is still warming up (len < 30), it will return 0.0
    assert vec[0] == 0.0
    
    # Let's inspect the z_ofi deque directly to verify raw value was 1.0
    assert engine.zs_ofi.deque[-1] == 1.0
    assert engine.zs_depth.deque[-1] == 0.0
    assert engine.zs_susc.deque[-1] == 0.0

def test_susceptibility_guard():
    engine = LiveFeatureEngine()
    # Init tick
    engine.compute_vector(2400.00, 2400.10, 5.0, 3.0, 1000)
    
    # Tick with zero volume (depth=0) but we pretend it's regular path (which should hit fallback actually)
    # Wait, if bid_vol <= 0, it hits fallback. Let's provide a tiny volume that might cause zero division
    # Actually, depth=0 triggers fallback. What if depth=1e-8?
    engine.compute_vector(2400.00, 2400.10, 1e-9, 1e-9, 1001)
    
    # No crash, z-score gets raw susc
    # Because raw_ofi will be 0
    import math
    assert not math.isnan(engine.zs_susc.deque[-1])

def test_weak_inequality():
    engine = LiveFeatureEngine()
    # Init tick
    engine.compute_vector(2400.00, 2400.10, 5.0, 3.0, 1000)
    
    # Tick with same price, changed volume (volume refresh)
    engine.compute_vector(2400.00, 2400.10, 10.0, 3.0, 1001)
    
    # dBid=0, dAsk=0
    # raw_ofi = (1)*10 - (1)*5 - (1)*3 + (1)*3 = 10 - 5 - 3 + 3 = 5.0
    assert engine.zs_ofi.deque[-1] == 5.0

def test_zscore_window():
    zs = RollingZScore(window=1000)
    
    import numpy as np
    
    # Push 1500 values
    np.random.seed(42)
    vals = np.random.randn(1500)
    
    for v in vals:
        z = zs.update(v)
        
    # Check if last z matches np.mean and np.std of last 1000
    last_1000 = vals[500:]
    mu = np.mean(last_1000)
    std = np.std(last_1000, ddof=1)
    
    assert abs(zs.mean - mu) < 1e-5
    assert abs(z - ((vals[-1] - mu) / std)) < 1e-5
