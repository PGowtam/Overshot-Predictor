"""
Phase 5 Verification: Ensemble Inference
Tests for majority voting and baiting logic.
"""
import unittest
from unittest.mock import MagicMock
from BrickOfTicks_Trader.inference.ensemble import EnsemblePredictor
from BrickOfTicks_Trader.config.settings import (
    BAIT_PROB_WIN_THRESHOLD, BAIT_PRED_OS_THRESHOLD
)

class TestEnsemblePredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = EnsemblePredictor("mock_dir")

    def _create_mock_model(self, prob_win, pred_os):
        """Creates a mock keras model that returns specified values."""
        mock_model = MagicMock()
        
        # We need mock_model([micro, macro], training=False) to return a structure
        # where preds[0].numpy().flatten()[0] == prob_win
        class FakeTensor:
            def __init__(self, val):
                self.val = val
            def numpy(self):
                return self
            def flatten(self):
                return [self.val]

        mock_model.return_value = [FakeTensor(prob_win), FakeTensor(pred_os)]
        return mock_model

    def test_voting_logic_all_signal(self):
        """All 3 models signal -> action 1."""
        self.predictor.models = [
            self._create_mock_model(0.8, 1.5),
            self._create_mock_model(0.8, 1.5),
            self._create_mock_model(0.8, 1.5),
        ]
        self.predictor.configs = [
            {"Prob_Win_threshold": 0.5, "Pred_OS_threshold": 1.2},
            {"Prob_Win_threshold": 0.5, "Pred_OS_threshold": 1.2},
            {"Prob_Win_threshold": 0.5, "Pred_OS_threshold": 1.2},
        ]
        
        res = self.predictor.predict("mock_micro", "mock_macro")
        self.assertEqual(res["action"], 1)
        self.assertEqual(res["votes"], 3)
        self.assertEqual(res["trade_type"], "standard")

    def test_voting_logic_two_signal(self):
        """2 models signal -> action 1."""
        self.predictor.models = [
            self._create_mock_model(0.8, 1.5),  # OK
            self._create_mock_model(0.8, 1.5),  # OK
            self._create_mock_model(0.4, 1.0),  # Not OK
        ]
        self.predictor.configs = [{"Prob_Win_threshold": 0.5, "Pred_OS_threshold": 1.2}] * 3
        
        res = self.predictor.predict("mock_micro", "mock_macro")
        self.assertEqual(res["action"], 1)
        self.assertEqual(res["votes"], 2)
        self.assertEqual(res["trade_type"], "standard")

    def test_voting_logic_one_signal(self):
        """1 model signals -> action 0."""
        self.predictor.models = [
            self._create_mock_model(0.8, 1.5),  # OK
            self._create_mock_model(0.4, 1.0),  # Not OK
            self._create_mock_model(0.4, 1.0),  # Not OK
        ]
        self.predictor.configs = [{"Prob_Win_threshold": 0.5, "Pred_OS_threshold": 1.2}] * 3
        
        res = self.predictor.predict("mock_micro", "mock_macro")
        self.assertEqual(res["action"], 0)
        self.assertEqual(res["votes"], 1)
        self.assertEqual(res["trade_type"], "none")

    def test_baiting_logic(self):
        """All 3 models predict high-confidence loss -> action -1 (REVERSE)."""
        self.predictor.models = [
            self._create_mock_model(BAIT_PROB_WIN_THRESHOLD - 0.1, BAIT_PRED_OS_THRESHOLD - 0.1),
            self._create_mock_model(BAIT_PROB_WIN_THRESHOLD - 0.1, BAIT_PRED_OS_THRESHOLD - 0.1),
            self._create_mock_model(BAIT_PROB_WIN_THRESHOLD - 0.1, BAIT_PRED_OS_THRESHOLD - 0.1),
        ]
        self.predictor.configs = [{"Prob_Win_threshold": 0.5, "Pred_OS_threshold": 1.2}] * 3
        
        res = self.predictor.predict("mock_micro", "mock_macro")
        self.assertEqual(res["action"], -1)
        self.assertEqual(res["votes"], 0)
        self.assertEqual(res["trade_type"], "baiting")

    def test_baiting_logic_incomplete(self):
        """Only 2 models predict high-confidence loss -> action 0 (no bait)."""
        self.predictor.models = [
            self._create_mock_model(BAIT_PROB_WIN_THRESHOLD - 0.1, BAIT_PRED_OS_THRESHOLD - 0.1),
            self._create_mock_model(BAIT_PROB_WIN_THRESHOLD - 0.1, BAIT_PRED_OS_THRESHOLD - 0.1),
            self._create_mock_model(BAIT_PROB_WIN_THRESHOLD + 0.1, BAIT_PRED_OS_THRESHOLD + 0.1),  # Fails bait check
        ]
        self.predictor.configs = [{"Prob_Win_threshold": 0.5, "Pred_OS_threshold": 1.2}] * 3
        
        res = self.predictor.predict("mock_micro", "mock_macro")
        self.assertEqual(res["action"], 0)
        self.assertEqual(res["trade_type"], "none")

if __name__ == '__main__':
    unittest.main()
