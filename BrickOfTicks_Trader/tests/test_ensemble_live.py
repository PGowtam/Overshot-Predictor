import os
import sys
import numpy as np
import pytest
from bridge.ensemble import EnsemblePredictor

try:
    import tensorflow as tf
except ImportError:
    tf = None

@pytest.mark.skipif(tf is None, reason="TensorFlow is not installed")
def test_ensemble_live_holdout_wr():
    """
    Live mathematical validation test.
    Loads real Keras models and real K=0.00295 holdout tensors to 
    assert that the unified PRED_OS_THRESHOLD=1.4 correctly reproduces 
    the ~90.3% win rate documented in the Offline Phase 9 tests.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback_dir = os.path.join(base_dir, "..", "outputs", "exec", "cv")
    
    ensemble = EnsemblePredictor(primary_dir=os.path.join(base_dir, "models"), 
                                 fallback_dir=fallback_dir)
    
    # Check if models exist before loading to prevent hanging CI if they are missing
    for i in range(1, 4):
        p1 = os.path.join(ensemble.primary_dir, f"fold_{i}", "model.keras")
        p2 = os.path.join(ensemble.fallback_dir, f"fold_{i}", "model.keras")
        if not os.path.exists(p1) and not os.path.exists(p2):
            pytest.skip(f"Models not found at {p1} or {p2}. Skipping live integration test.")
            
    # Load Models
    ensemble.load()
    
    # Load Tensors
    tensor_dir = os.path.join(base_dir, "..", "outputs", "tensors_holdout_K295")
    if not os.path.exists(tensor_dir):
        pytest.skip("Holdout tensors not found. Skipping live test.")
        
    micro = np.load(os.path.join(tensor_dir, "holdout_micro.npy"))
    macro = np.load(os.path.join(tensor_dir, "holdout_macro.npy"))
    y_class = np.load(os.path.join(tensor_dir, "holdout_y_class.npy"))
    
    # Evaluate
    trades_taken = 0
    wins = 0
    
    for i in range(len(micro)):
        # Provide batch dim
        m_in = micro[i:i+1]
        c_in = macro[i:i+1]
        
        result = ensemble.predict(m_in, c_in)
        
        if result['action'] == 1:
            trades_taken += 1
            # y_class indicates true label: 1=Win, 0=Loss
            if y_class[i] == 1:
                wins += 1
                
    if trades_taken == 0:
        pytest.fail("Ensemble took 0 trades on holdout. Threshold might be broken.")
        
    win_rate = (wins / trades_taken) * 100
    print(f"\nLive Test Results: {trades_taken} Trades, {wins} Wins, {win_rate:.2f}% WR")
    
    # Assert WR is within 5% absolute of 90.3%
    assert abs(win_rate - 90.3) < 5.0, f"Win rate {win_rate:.2f}% deviated heavily from 90.3% target!"
