"""
Execution Models (Dual-Model Architecture)
==========================================
Defines the baseline execution model and the Markov Chain experimental model.
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D, MaxPool1D, Dense, Flatten, Dropout, LSTM,
    Concatenate, TimeDistributed, LeakyReLU, Reshape
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
    
    b1 = build_cnn_branch(tick_input, kernel_size=1)
    b3 = build_cnn_branch(tick_input, kernel_size=3)
    b5 = build_cnn_branch(tick_input, kernel_size=5)
    
    merged = Concatenate()([b1, b3, b5])
    pooled = MaxPool1D(pool_size=4)(merged)
    flat = Flatten()(pooled)
    
    dense = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(flat)
    out = Dropout(0.3)(dense)
    
    return Model(inputs=tick_input, outputs=out, name="cnn_encoder")


def build_baseline_exec_model():
    """Builds Model A: Baseline Execution Model (Dual-Head CNN+LSTM)."""
    micro_input = Input(shape=(10, 100, 9), name='micro_input')
    macro_input = Input(shape=(10, 3), name='macro_input')
    
    cnn_encoder = build_cnn_encoder(input_shape=(100, 9))
    cnn_output = TimeDistributed(cnn_encoder)(micro_input)
    
    fused = Concatenate(axis=-1)([cnn_output, macro_input])
    
    lstm_out = LSTM(32, return_sequences=False, kernel_regularizer=l2(1e-4))(fused)
    lstm_out = Dropout(0.3)(lstm_out)
    
    prob_win = Dense(1, activation='sigmoid', name='prob_win')(lstm_out)
    pred_os = Dense(1, activation='relu', name='pred_os')(lstm_out)
    
    model = Model(inputs=[micro_input, macro_input],
                  outputs=[prob_win, pred_os],
                  name="baseline_exec_model")
    return model


def build_markov_exec_model():
    """Builds Model B: Markov Chain Execution Model (Model A + Sequence Branch)."""
    micro_input = Input(shape=(10, 100, 9), name='micro_input')
    macro_input = Input(shape=(10, 3), name='macro_input')
    seq_input = Input(shape=(100,), name='seq_input')
    
    # ── Standard CNN+LSTM Path ──
    cnn_encoder = build_cnn_encoder(input_shape=(100, 9))
    cnn_output = TimeDistributed(cnn_encoder)(micro_input)
    fused_macro = Concatenate(axis=-1)([cnn_output, macro_input])
    lstm_out = LSTM(32, return_sequences=False, kernel_regularizer=l2(1e-4))(fused_macro)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # ── Markov Sequence Path ──
    # The sequence is 100 binary digits representing past brick directions.
    # Reshape to (100, 1) for Conv1D
    seq_reshaped = Reshape((100, 1))(seq_input)
    seq_conv1 = Conv1D(filters=8, kernel_size=3, activation='relu')(seq_reshaped)
    seq_pool1 = MaxPool1D(pool_size=4)(seq_conv1)
    seq_conv2 = Conv1D(filters=16, kernel_size=3, activation='relu')(seq_pool1)
    seq_pool2 = MaxPool1D(pool_size=4)(seq_conv2)
    seq_flat = Flatten()(seq_pool2)
    seq_dense = Dense(16, activation='relu')(seq_flat)
    
    # ── Final Fusion ──
    final_fused = Concatenate()([lstm_out, seq_dense])
    final_fused = Dropout(0.2)(final_fused)
    
    prob_win = Dense(1, activation='sigmoid', name='prob_win')(final_fused)
    pred_os = Dense(1, activation='relu', name='pred_os')(final_fused)
    
    model = Model(inputs=[micro_input, macro_input, seq_input],
                  outputs=[prob_win, pred_os],
                  name="markov_exec_model")
    return model


def compile_model(model):
    """Compile model with dual-head loss."""
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
