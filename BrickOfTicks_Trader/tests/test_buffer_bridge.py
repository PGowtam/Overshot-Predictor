import pytest
import numpy as np
from bridge.buffer import InferenceBuffer

def test_buffer_initialization():
    buf = InferenceBuffer()
    assert buf.micro.maxlen == 100
    assert buf.snapshots.maxlen == 10
    assert buf.macro.maxlen == 10

def test_zero_padding_and_rewrites():
    buf = InferenceBuffer()
    
    # Push 50 ticks, all part of brick ID 1
    for _ in range(50):
        vec = [1.0] * 9
        buf.append_tick(vec, 1)
        
    # Simulate a brick close at brick ID 1
    macro = [0.1, 1.0, -0.5]
    res = buf.on_brick_close(1, macro)
    
    assert res is None  # Since we don't have 10 snapshots yet
    assert len(buf.snapshots) == 1
    
    snap = buf.snapshots[-1]
    
    # First 50 rows should be entirely zero (padding)
    assert np.all(snap[:50] == 0.0)
    
    # Last 50 rows should have the rewritten values
    # Original was all 1.0, but:
    # vec[6] (Flag_Curr) should be 1.0 (since b_id 1 == current_brick_id 1)
    # vec[8] (Decay) should be 0.0 ((1 - 1) / 100)
    assert snap[50, 6] == 1.0
    assert snap[50, 8] == 0.0
    assert snap[50, 0] == 1.0 # First feature remains 1.0

def test_decay_computation():
    buf = InferenceBuffer()
    
    # Tick from brick 1
    buf.append_tick([0.0]*9, 1)
    
    # Brick 5 closes
    buf.on_brick_close(5, [0.0]*3)
    
    snap = buf.snapshots[-1]
    # The tick is at the last row (index 99)
    # Decay = (5 - 1) / 100 = 0.04
    assert snap[-1, 8] == 0.04
    # Flag_curr = 0.0 (since 1 != 5)
    assert snap[-1, 6] == 0.0

def test_ten_bricks_gate():
    buf = InferenceBuffer()
    
    # Push 9 bricks
    for i in range(1, 10):
        buf.append_tick([0.0]*9, i)
        res = buf.on_brick_close(i, [0.0]*3)
        assert res is None
        
    # Push 10th brick
    buf.append_tick([0.0]*9, 10)
    res = buf.on_brick_close(10, [0.0]*3)
    
    assert res is not None
    micro_tensor, macro_tensor = res
    
    assert micro_tensor.shape == (1, 10, 100, 9)
    assert macro_tensor.shape == (1, 10, 3)
