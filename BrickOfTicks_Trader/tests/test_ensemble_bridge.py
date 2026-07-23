import pytest
import numpy as np
from bridge.ensemble import EnsemblePredictor

class MockTensor:
    def __init__(self, value):
        self.value = value
    def numpy(self):
        return np.array([[self.value]])

class MockModel:
    def __init__(self, prob_win: float, pred_os: float):
        self.prob_win = prob_win
        self.pred_os = pred_os

    def __call__(self, inputs, training=False):
        # Return mock predictions as MockTensors
        return [MockTensor(self.prob_win), MockTensor(self.pred_os)]

def test_ensemble_constants():
    assert EnsemblePredictor.PRED_OS_THRESHOLD == 1.4
    assert EnsemblePredictor.PROB_WIN_THRESHOLD == 0.5
    assert EnsemblePredictor.VOTE_THRESHOLD == 2

def test_no_baiting_action():
    ensemble = EnsemblePredictor()
    
    # Mock models that output garbage under the threshold
    ensemble.models = [
        MockModel(prob_win=0.05, pred_os=0.3),
        MockModel(prob_win=0.10, pred_os=0.4),
        MockModel(prob_win=0.15, pred_os=0.5)
    ]
    
    micro = np.zeros((1, 10, 100, 9))
    macro = np.zeros((1, 10, 3))
    
    result = ensemble.predict(micro, macro)
    
    # The result action should NEVER be -1. It should strictly be 0.
    assert result['action'] == 0
    assert result['votes'] == 0

def test_ensemble_voting_success():
    ensemble = EnsemblePredictor()
    
    # Mock models: 2 out of 3 pass the threshold
    # Thresholds are >= 0.5 and >= 1.4
    ensemble.models = [
        MockModel(prob_win=0.51, pred_os=1.45), # Vote YES
        MockModel(prob_win=0.49, pred_os=1.50), # Vote NO (prob_win too low)
        MockModel(prob_win=0.80, pred_os=1.41)  # Vote YES
    ]
    
    micro = np.zeros((1, 10, 100, 9))
    macro = np.zeros((1, 10, 3))
    
    result = ensemble.predict(micro, macro)
    
    assert result['votes'] == 2
    assert result['action'] == 1  # Since 2 >= VOTE_THRESHOLD
    assert result['details'][0]['signal'] is True
    assert result['details'][1]['signal'] is False
    assert result['details'][2]['signal'] is True
