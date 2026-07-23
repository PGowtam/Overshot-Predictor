import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from collections import deque
import pickle
import os
from pathlib import Path

from BrickOfTicks_Trader.utils.state import StateManager
from BrickOfTicks_Trader.config.settings import LOGS_DIR

class DummyBuffer:
    def __init__(self):
        self.micro_buffer = deque([(np.array([1,2,3]), 1)])
        self.macro_history = deque([np.array([4,5,6])])
        self.snapshots = deque([(np.array([7,8,9]), 1000)])
        self.brick_size_history = [1.0]
        self.current_brick_id = 1

class DummyFeatures:
    def __init__(self):
        self.z_ofi = MagicMock()
        self.z_ofi.deque = deque([0.1])
        self.z_depth = MagicMock()
        self.z_depth.deque = deque([0.2])
        self.z_susc = MagicMock()
        self.z_susc.deque = deque([0.3])
        self.z_vel = MagicMock()
        self.z_vel.deque = deque([0.4])
        self.z_spread = MagicMock()
        self.z_spread.deque = deque([0.5])
        
        self.prev_bid = 1.0
        self.prev_ask = 1.1
        self.prev_mid = 1.05
        self.current_brick_open = 1.0
        self.current_brick_size = 1.0
        self.current_brick_id = 1

class DummyRenko:
    def __init__(self):
        self.current_price = 100.0
        self.uptrend = 1
        self.sequence = "101"
        self.brick_size = 1.0
        self.history = [1]

class TestStartupPersistence(unittest.TestCase):
    
    def setUp(self):
        self.state = StateManager()
        self.pkl_path = Path(LOGS_DIR) / "internal_state.pkl"
        if self.pkl_path.exists():
            os.remove(self.pkl_path)
    
    def tearDown(self):
        if self.pkl_path.exists():
            os.remove(self.pkl_path)
            
    def test_save_and_load_persistence(self):
        # 1. Create dummies
        d_buf = DummyBuffer()
        d_feat = DummyFeatures()
        d_renko = DummyRenko()
        
        # 2. Save
        self.state.save_internal_state(d_feat, d_buf, d_renko)
        self.assertTrue(self.pkl_path.exists())
        
        # 3. Create fresh dummies
        new_buf = DummyBuffer()
        new_buf.micro_buffer.clear()
        new_buf.macro_history.clear()
        new_buf.snapshots.clear()
        
        new_feat = DummyFeatures()
        new_feat.z_ofi.deque.clear()
        new_feat.z_depth.deque.clear()
        
        new_renko = DummyRenko()
        new_renko.current_price = 0.0
        
        # 4. Load
        res = self.state.load_internal_state(new_feat, new_buf, new_renko)
        self.assertTrue(res)
        
        # 5. Verify Parity
        self.assertEqual(new_renko.current_price, 100.0)
        self.assertEqual(new_renko.sequence, "101")
        
        self.assertTrue(np.array_equal(new_buf.micro_buffer[0][0], np.array([1,2,3])))
        self.assertEqual(list(new_feat.z_ofi.deque)[0], 0.1)

if __name__ == '__main__':
    unittest.main()
