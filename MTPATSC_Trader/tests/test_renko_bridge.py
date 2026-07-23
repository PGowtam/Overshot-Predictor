import pytest
from bridge.renko import RenkoBuilder, K_MULTIPLIER

def test_renko_k_multiplier():
    assert K_MULTIPLIER == 0.00295, "CRITICAL: K_MULTIPLIER must be 0.00295"

def test_brick_size_calculation():
    day_open = 2400.00
    builder = RenkoBuilder(day_open)
    expected_size = 2400.00 * 0.00295
    assert abs(builder.brick_size - expected_size) < 1e-5
    assert abs(builder.brick_size - 7.08) < 1e-5

def test_up_brick_formation():
    builder = RenkoBuilder(2400.00)
    bs = builder.brick_size # 7.08
    
    # Tick below brick size - no brick
    bricks = builder.update_tick(2405.00, 1000)
    assert len(bricks) == 0
    
    # Tick reaches 1st brick size
    bricks = builder.update_tick(2407.08, 1001)
    assert len(bricks) == 1
    b = bricks[0]
    assert b.open == 2400.00
    assert b.close == 2407.08
    assert b.uptrend == 1
    assert builder.sequence == '1'

def test_reversal_down():
    builder = RenkoBuilder(2400.00)
    bs = builder.brick_size # 7.08
    
    # Make one UP brick
    builder.update_tick(2407.08, 1000)
    
    # Drop by 1 brick (no reversal yet)
    bricks = builder.update_tick(2400.00, 1001)
    assert len(bricks) == 0
    
    # Drop by 2x brick (reversal!)
    bricks = builder.update_tick(2392.92, 1002)
    assert len(bricks) == 1
    b = bricks[0]
    assert b.open == 2400.00
    assert b.close == 2392.92
    assert b.uptrend == -1
    assert builder.sequence == '10'

def test_gap_fill():
    builder = RenkoBuilder(2400.00)
    bs = builder.brick_size # 7.08
    
    # Jump by 5 bricks up
    target_price = 2400.00 + 5 * bs
    bricks = builder.update_tick(target_price + 0.1, 1000)
    assert len(bricks) == 5
    assert builder.sequence == '11111'
    
    # Check contiguity
    for i in range(5):
        assert abs(bricks[i].close - (2400.00 + (i+1)*bs)) < 1e-5
        if i > 0:
            assert abs(bricks[i].open - bricks[i-1].close) < 1e-5

def test_extreme_volatility():
    builder = RenkoBuilder(2400.00)
    bs = builder.brick_size # 7.08
    
    # Volatility sequence
    builder.update_tick(2400.00 + 2*bs, 1000) # 2 UP
    assert builder.brick_count == 2
    assert builder.sequence == '11'
    
    # Reverse down 3 bricks (needs 2x reversal + 1 gap)
    target = 2400.00 + 2*bs - 4*bs # net -2bs from open
    bricks = builder.update_tick(target, 1001)
    assert len(bricks) == 3
    assert builder.sequence == '11000'
    assert bricks[0].uptrend == -1
    assert abs(bricks[-1].close - target) < 1e-5

def test_update_brick_size():
    builder = RenkoBuilder(2400.00)
    assert abs(builder.brick_size - 7.08) < 1e-5
    
    # Rollover to 2500
    expected = 2500.00 * 0.00295 # 7.375
    builder.update_brick_size(expected)
    assert abs(builder.brick_size - expected) < 1e-5
    # Current price remains 2400.00
    assert builder.current_price == 2400.00
