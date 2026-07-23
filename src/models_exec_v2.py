"""
Execution Models V2 (Transformer Architecture)
==============================================
Implements the Selective Trade Discovery System network architecture, featuring:
1. Micro-Branch: 4-layer Transformer Encoder (100 ticks -> 128 embedding)
2. Macro-Branch: 2-layer Transformer Encoder (10 bricks -> 64 embedding)
3. Summary-Branch: 5D vector
4. Fusion Layer & Shared Trunk
5. Dual Heads (prob_win, pred_os)
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, Dropout, Concatenate, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D, Layer, Embedding, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber

@tf.keras.utils.register_keras_serializable()
class PositionEmbedding(Layer):
    """Adds learned positional embeddings to the sequence."""
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class TransformerEncoderBlock(Layer):
    """Transformer Encoder with Pre-LN (norm_first=True) architecture."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        # Pre-LN architecture (norm_first=True)
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        
        x2 = self.layernorm2(out1)
        ffn_output = self.ffn(x2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class ResidualBlock(Layer):
    """Residual block for the shared trunk."""
    def __init__(self, out_dim, rate=0.3, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.rate = rate
        
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.dense1 = Dense(out_dim)
        self.gelu = Activation('gelu')
        self.dropout = Dropout(rate)
        self.dense2 = Dense(out_dim)
        self.proj = None
        
    def build(self, input_shape):
        in_dim = input_shape[-1]
        if in_dim != self.out_dim:
            self.proj = Dense(self.out_dim) # Linear projection for skip connection

    def call(self, inputs, training=False):
        x = self.layernorm(inputs)
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        
        shortcut = inputs
        if self.proj is not None:
            shortcut = self.proj(shortcut)
            
        return shortcut + x

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_dim": self.out_dim,
            "rate": self.rate,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class BinaryFocalCrossentropy(tf.keras.losses.Loss):
    """Focal Loss with alpha balancing."""
    def __init__(self, alpha=0.35, gamma=2.0, name='binary_focal_crossentropy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        bce = -tf.math.log(p_t)
        focal_loss = alpha_factor * modulating_factor * bce
        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
        })
        return config

def build_transformer_exec_model():
    """Builds the complete dual-branch Transformer execution model."""
    
    # ── Inputs ──
    micro_input = Input(shape=(100, 9), name='micro_input')
    macro_input = Input(shape=(10, 11), name='macro_input')
    summary_input = Input(shape=(5,), name='summary_input')
    
    # ── Micro Branch (4-Layer Transformer) ──
    # Linear Projection: 9 -> 128
    micro_proj = Dense(128, use_bias=False)(micro_input)
    micro_proj = LayerNormalization(epsilon=1e-6)(micro_proj)
    
    x_micro = PositionEmbedding(maxlen=100, embed_dim=128)(micro_proj)
    for _ in range(4):
        x_micro = TransformerEncoderBlock(embed_dim=128, num_heads=8, ff_dim=512, rate=0.1)(x_micro)
        
    # Mean-pooling across positions
    micro_embed = GlobalAveragePooling1D()(x_micro) # (batch, 128)
    
    # ── Macro Branch (2-Layer Transformer) ──
    # Linear Projection: 11 -> 64
    macro_proj = Dense(64, use_bias=False)(macro_input)
    macro_proj = LayerNormalization(epsilon=1e-6)(macro_proj)
    
    x_macro = PositionEmbedding(maxlen=10, embed_dim=64)(macro_proj)
    for _ in range(2):
        x_macro = TransformerEncoderBlock(embed_dim=64, num_heads=4, ff_dim=256, rate=0.1)(x_macro)
        
    # Mean-pooling across positions
    macro_embed = GlobalAveragePooling1D()(x_macro) # (batch, 64)
    
    # ── Fusion Layer ──
    # Concatenate [MicroEmbed(128), MacroEmbed(64), Summary(5)] -> 197
    fused = Concatenate()([micro_embed, macro_embed, summary_input])
    
    # ── Shared Trunk ──
    # 197 -> 256 -> 128 -> 64
    trunk = ResidualBlock(out_dim=256, rate=0.3)(fused)
    trunk = ResidualBlock(out_dim=128, rate=0.3)(trunk)
    trunk = ResidualBlock(out_dim=64, rate=0.3)(trunk)
    
    # ── Output Heads ──
    prob_win = Dense(1, activation='sigmoid', name='prob_win')(trunk)
    
    # Overshoot Regression (softplus to ensure positive but without hard zeroing small overshots)
    pred_os = Dense(1, activation='softplus', name='pred_os')(trunk)
    
    model = Model(inputs=[micro_input, macro_input, summary_input],
                  outputs=[prob_win, pred_os],
                  name="transformer_exec_model_v2")
    return model


def compile_transformer_model(model):
    """Compile model with AdamW, standard LR and multiple loss heads."""
    model.compile(
        optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4, clipnorm=1.0),
        loss={
            'prob_win': BinaryFocalCrossentropy(alpha=0.35, gamma=2.0),
            'pred_os': Huber(delta=0.5)
        },
        loss_weights={
            'prob_win': 0.5,
            'pred_os': 0.3
        },
        metrics={
            'prob_win': 'accuracy',
            'pred_os': 'mae'
        }
    )
    return model
