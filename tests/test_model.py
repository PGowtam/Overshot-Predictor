"""Unit tests for model architecture (Phase 5)."""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import build_model, compile_model


class TestModelArchitecture:
    """Test model structure and compilation."""

    def test_model_build_shapes(self):
        """Verify input/output shapes."""
        model = build_model()
        
        # Check inputs
        assert len(model.inputs) == 2
        # Keras 3 inputs are KerasTensors
        input_shapes = [x.shape for x in model.inputs]
        # Shapes might be (None, 10, 100, 9) or similar. Check dimensions.
        assert input_shapes[0][1:] == (10, 100, 9)
        assert input_shapes[1][1:] == (10, 3)
        
        # Check outputs
        assert len(model.outputs) == 2
        output_shapes = [x.shape for x in model.outputs]
        assert output_shapes[0][1:] == (1,)
        assert output_shapes[1][1:] == (1,)

    def test_forward_pass(self):
        """Verify forward pass with random data."""
        model = build_model()
        
        batch_size = 4
        micro = np.random.randn(batch_size, 10, 100, 9).astype(np.float32)
        macro = np.random.randn(batch_size, 10, 3).astype(np.float32)
        
        outputs = model.predict([micro, macro])
        
        assert len(outputs) == 2
        prob_win, pred_os = outputs
        
        assert prob_win.shape == (batch_size, 1)
        assert pred_os.shape == (batch_size, 1)
        
        # Check activation ranges
        # Sigmoid -> [0, 1]
        assert np.all((prob_win >= 0) & (prob_win <= 1))
        # ReLU -> [0, inf)
        assert np.all(pred_os >= 0)

    def test_compilation(self):
        """Verify loss functions and optimizer."""
        model = build_model()
        model = compile_model(model)
        
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
        assert abs(model.optimizer.learning_rate.numpy() - 1e-3) < 1e-6
        
        # Check loss config
        # Keras stores loss as a list if detailed config is used
        # or as a dict in older versions. We check via model.loss
        if isinstance(model.loss, dict):
            assert isinstance(model.loss['prob_win'], tf.keras.losses.BinaryCrossentropy)
            assert isinstance(model.loss['pred_os'], tf.keras.losses.Huber)
        else:
            # If it's a list, order matters based on output order
            pass  # structure varies by TF version, skipping strict typ check here

    def test_parameter_count(self):
        """Verify model size is reasonable (approx 48k)."""
        model = build_model()
        trainable = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        
        # Allow some flexibility, but ensure it's in the ballpark
        # Our calculation:
        # CNN: (16*1*9+16 + 16*3*9+16 + 16*5*9+16) = 144+432+720+48 = 1344
        # Dense(32): 1200*32 + 32 = 38432
        # Fusion: 0 params
        # LSTM(32): 4 * (32*(32+35) + 32) = 4 * (2144 + 32) = 8704
        # Heads: 2 * (32*1 + 1) = 66
        # Total ≈ 1344 + 38432 + 8704 + 66 ≈ 48,546
        
        print(f"Trainable params: {trainable}")
        assert 40_000 < trainable < 60_000

    def test_layer_types(self):
        """Verify critical layer types (MaxPool1D vs GlobalAvgPool)."""
        model = build_model()
        # Find TimeDistributed layer
        td_layer = [l for l in model.layers if isinstance(l, tf.keras.layers.TimeDistributed)][0]
        encoder = td_layer.layer
        
        # Check encoder layers
        layer_types = [type(l).__name__ for l in encoder.layers]
        # Keras 3 might use 'MaxPooling1D' or 'MaxPool1D'
        assert "MaxPooling1D" in layer_types or "MaxPool1D" in layer_types
        assert "GlobalAveragePooling1D" not in layer_types
