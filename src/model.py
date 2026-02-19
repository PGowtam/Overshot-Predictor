"""
Phase 5: Model Architecture (FR-ML-01 to FR-ML-04)

Dual-head CNN+LSTM model for predicting trade outcome (WIN/LOSS) and magnitude (overshoot).

Architecture:
  1. Inputs: 
     - Micro: (Batch, 10, 100, 9)
     - Macro: (Batch, 10, 3)
     
  2. TimeDistributed CNN Block (shared weights across 10 steps):
     - 3 Parallel Conv1D branches (k=1,3,5), 16 filters, causal padding
     - Concatenate → MaxPool1D(4) → Flatten
     - Dense(32, relu) → Dropout(0.3)
     - Output: (Batch, 10, 32)
     
  3. Fusion:
     - Concat CNN output (10, 32) + Macro input (10, 3) → (Batch, 10, 35)
     
  4. LSTM:
     - LSTM(32) → Dropout(0.3) → (Batch, 32)
     
  5. Dual Heads:
     - Head A (prob_win): Dense(1, sigmoid)
     - Head B (pred_os): Dense(1, relu)
     
Compilation:
  - Optimizer: Adam(lr=1e-3)
  - Loss: BCE (Head A) + 0.3 * Huber (Head B)
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D, MaxPool1D, Dense, Flatten, Dropout, LSTM,
    Concatenate, TimeDistributed, LeakyReLU, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, Huber


def build_cnn_branch(x, kernel_size, filters=16, alpha=0.1):
    """Single Conv1D branch with LeakyReLU activation."""
    x = Conv1D(filters=filters, kernel_size=kernel_size,
               padding='causal', activation=None)(x)
    return LeakyReLU(negative_slope=alpha)(x)


def build_cnn_encoder(input_shape=(100, 9), l2_reg=1e-4):
    """Build the CNN encoder model to be wrapped in TimeDistributed."""
    
    tick_input = Input(shape=input_shape)
    
    # --- Parallel Branches ---
    # Branch 1 (k=1): Instantaneous features
    b1 = build_cnn_branch(tick_input, kernel_size=1)
    
    # Branch 2 (k=3): Short-term patterns
    b3 = build_cnn_branch(tick_input, kernel_size=3)
    
    # Branch 3 (k=5): Medium-term patterns
    b5 = build_cnn_branch(tick_input, kernel_size=5)
    
    # --- Merge & Pool ---
    # Concatenate along channel axis: (100, 16*3) = (100, 48)
    merged = Concatenate()([b1, b3, b5])
    
    # MaxPool: (100, 48) -> (25, 48)
    # We use size=4 to reduce 100 ticks to 25 feature maps
    pooled = MaxPool1D(pool_size=4)(merged)
    
    # --- Embed ---
    flat = Flatten()(pooled)  # (25 * 48) = 1200
    
    dense = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(flat)
    out = Dropout(0.3)(dense)
    
    return Model(inputs=tick_input, outputs=out, name="cnn_encoder")


def build_model():
    """Build the full dual-head CNN+LSTM model."""
    
    # ── Inputs ──────────────────────────────────────────────────
    # Micro: 10 bricks of history, each 100 ticks, 9 features
    micro_input = Input(shape=(10, 100, 9), name='micro_input')
    
    # Macro: 10 bricks of history, 3 features
    macro_input = Input(shape=(10, 3), name='macro_input')
    
    # ── TimeDistributed CNN ─────────────────────────────────────
    # Create the encoder model once
    cnn_encoder = build_cnn_encoder(input_shape=(100, 9))
    
    # Apply to each of the 10 timesteps
    # Input: (Batch, 10, 100, 9) -> Output: (Batch, 10, 32)
    cnn_output = TimeDistributed(cnn_encoder)(micro_input)
    
    # ── Fusion ──────────────────────────────────────────────────
    # Concatenate CNN output (32) with Macro input (3) along feature axis
    # (Batch, 10, 32) + (Batch, 10, 3) -> (Batch, 10, 35)
    fused = Concatenate(axis=-1)([cnn_output, macro_input])
    
    # ── LSTM ────────────────────────────────────────────────────
    # Process sequence of 10 bricks
    lstm_out = LSTM(32, return_sequences=False,
                    kernel_regularizer=l2(1e-4))(fused)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # ── Dual Heads ──────────────────────────────────────────────
    # Head A: Probability of Win (Binary Classification)
    prob_win = Dense(1, activation='sigmoid', name='prob_win')(lstm_out)
    
    # Head B: Predicted Overshoot Magnitude (Regression, ReLU > 0)
    pred_os = Dense(1, activation='relu', name='pred_os')(lstm_out)
    
    # ── Model ───────────────────────────────────────────────────
    model = Model(inputs=[micro_input, macro_input],
                  outputs=[prob_win, pred_os],
                  name="overshot_predictor")
    
    return model


def compile_model(model):
    """Compile model with dual-head loss and specific weights."""
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={
            'prob_win': BinaryCrossentropy(),
            'pred_os': Huber(delta=1.0)
        },
        loss_weights={
            'prob_win': 1.0,
            'pred_os': 0.3
        },
        metrics={
            'prob_win': 'accuracy',
            'pred_os': 'mae'
        }
    )
    return model


if __name__ == "__main__":
    model = build_model()
    model = compile_model(model)
    model.summary()
