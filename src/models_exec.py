"""
Execution Models (Dual-Model Architecture)
==========================================
Defines the baseline execution model and the Markov Chain experimental model.
Uses the same proven architecture pattern from model.py.
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D, MaxPool1D, Dense, Flatten, Dropout, LSTM,
    Concatenate, TimeDistributed, LeakyReLU, Reshape,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, Huber, BinaryFocalCrossentropy


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
    """Builds Model A: Baseline Execution Model (Dual-Head CNN+LSTM).
    
    Same architecture as model.py build_model() — proven to train on M4 Mac.
    """
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


def build_bucketized_exec_model():
    """Builds Model B: Bucketized Execution Model (EXP-00B).
    
    Same architecture as baseline, but with a 4-class softmax head for pred_os.
    """
    micro_input = Input(shape=(10, 100, 9), name='micro_input')
    macro_input = Input(shape=(10, 3), name='macro_input')
    
    cnn_encoder = build_cnn_encoder(input_shape=(100, 9))
    cnn_output = TimeDistributed(cnn_encoder)(micro_input)
    
    fused = Concatenate(axis=-1)([cnn_output, macro_input])
    
    lstm_out = LSTM(32, return_sequences=False, kernel_regularizer=l2(1e-4))(fused)
    lstm_out = Dropout(0.3)(lstm_out)
    
    prob_win = Dense(1, activation='sigmoid', name='prob_win')(lstm_out)
    pred_os_class = Dense(4, activation='softmax', name='pred_os_class')(lstm_out)
    
    model = Model(inputs=[micro_input, macro_input], 
                  outputs=[prob_win, pred_os_class], 
                  name="bucketized_exec_model")
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

# High-MI dims identified from analysis
HIGH_MI_DIMS = [0, 9, 11, 12, 13, 21, 23, 28, 29]

def build_cnn_encoder_v2(input_shape=(100, 9), l2_reg=1e-4):
    """Wider CNN with second conv layer."""
    tick_input = Input(shape=input_shape)
    
    def cnn_branch(x, k, f=32):
        x = Conv1D(filters=f, kernel_size=k, padding='causal')(x)
        return LeakyReLU(negative_slope=0.1)(x)
    
    b1  = cnn_branch(tick_input, k=1,  f=32)
    b3  = cnn_branch(tick_input, k=3,  f=32)
    b5  = cnn_branch(tick_input, k=5,  f=32)
    b10 = cnn_branch(tick_input, k=10, f=16)
    
    merged = Concatenate()([b1, b3, b5, b10])     # (100, 112)
    pooled = MaxPool1D(pool_size=4)(merged)         # (25, 112)
    
    conv2  = Conv1D(64, kernel_size=3, padding='causal', activation='relu')(pooled)
    pooled2 = MaxPool1D(pool_size=4)(conv2)         # (6, 64)
    
    flat  = Flatten()(pooled2)
    dense = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(flat)
    out   = Dropout(0.3)(dense)
    
    return Model(inputs=tick_input, outputs=out, name="cnn_encoder_v2")

def _mi_bias_initializer(high_mi_dims, boost=2.0):
    """
    Returns an initializer that sets bias high for known high-MI dimensions.
    Gate sigmoid(bias) ≈ 0.88 for high-MI dims, 0.50 for others.
    """
    def initializer(shape, dtype=None):
        bias = tf.zeros(shape)
        indices = tf.constant([[d] for d in high_mi_dims])
        updates = tf.constant([boost] * len(high_mi_dims), dtype=tf.float32)
        bias = tf.tensor_scatter_nd_update(bias, indices, updates)
        return bias
    return initializer

def build_attention_exec_model(high_mi_dims=HIGH_MI_DIMS):
    """
    V2 model with:
    1. Wider CNN encoder
    2. MI-guided feature gating before attention
    3. Multi-head self-attention over brick sequence
    4. Dual LSTM path for recency
    """
    micro_input = Input(shape=(10, 100, 9), name='micro_input')
    macro_input = Input(shape=(10, 7),      name='macro_input')
    
    # CNN Encoding
    cnn_encoder = build_cnn_encoder_v2(input_shape=(100, 9))
    cnn_out = TimeDistributed(cnn_encoder)(micro_input)  # (batch, 10, 32)
    
    # MI-Guided Feature Gate
    gate = Dense(32, activation='sigmoid',
                 bias_initializer=_mi_bias_initializer(high_mi_dims),
                 name='mi_gate')(cnn_out)               # (batch, 10, 32)
    gated_cnn = cnn_out * gate                          # (batch, 10, 32)
    
    # Fuse with Macro
    fused = Concatenate(axis=-1)([gated_cnn, macro_input])  # (batch, 10, 35)
    fused = LayerNormalization()(fused)
    
    # Multi-Head Self-Attention over brick sequence
    attn_out = MultiHeadAttention(
        num_heads=4, key_dim=8, dropout=0.1
    )(fused, fused)                                     # (batch, 10, 35)
    attn_out = LayerNormalization()(attn_out + fused)   # residual
    
    from tensorflow.keras.layers import Softmax, Lambda
    
    # Dual path: attention pooling + LSTM recency
    attn_scores  = Dense(1, activation='tanh')(attn_out)          # (batch, 10, 1)
    attn_weights = Softmax(axis=1)(attn_scores)                   # (batch, 10, 1)
    context      = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([attn_weights, attn_out]) # (batch, 35)
    
    lstm_out = LSTM(32, return_sequences=False,
                    kernel_regularizer=l2(1e-4))(attn_out)        # (batch, 32)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # Final fusion and heads
    combined = Concatenate()([context, lstm_out])                 # (batch, 67)
    combined = Dense(48, activation='relu',
                     kernel_regularizer=l2(1e-4))(combined)
    combined = Dropout(0.2)(combined)
    
    prob_win = Dense(1, activation='sigmoid', name='prob_win')(combined)
    pred_os  = Dense(1, activation='relu',    name='pred_os')(combined)
    
    return Model(inputs=[micro_input, macro_input],
                 outputs=[prob_win, pred_os],
                 name="attention_exec_model_v2")
