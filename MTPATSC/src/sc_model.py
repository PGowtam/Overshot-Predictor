import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout, Concatenate, TimeDistributed, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_setup_classifier(l2_reg: float = 1e-4) -> Model:
    # ── Inputs ──────────────────────────────────────────────────
    ancs_fine_input    = Input(shape=(10, 6), name='ancs_fine')
    ancs_coarse_input  = Input(shape=(5, 6),  name='ancs_coarse')
    history_input      = Input(shape=(5, 5, 6), name='history')
    scalar_input       = Input(shape=(34,),   name='scalars')
    
    # ── Branch A: Fine ANCS ─────────────────────────────────────
    # Captures micro-patterns within the current brick
    a = Conv1D(32, kernel_size=1, activation='relu',
               padding='causal', kernel_regularizer=l2(l2_reg))(ancs_fine_input)
    a = Conv1D(32, kernel_size=3, activation='relu',
               padding='causal', kernel_regularizer=l2(l2_reg))(a)
    a = Flatten()(a)
    a = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(a)
    a = Dropout(0.3)(a)
    
    # ── Branch B: Coarse ANCS ───────────────────────────────────
    # Captures higher-level shape within the current brick
    b = Conv1D(16, kernel_size=1, activation='relu',
               padding='causal', kernel_regularizer=l2(l2_reg))(ancs_coarse_input)
    b = Flatten()(b)
    b = Dense(16, activation='relu', kernel_regularizer=l2(l2_reg))(b)
    b = Dropout(0.3)(b)
    
    # ── Branch C: History Context ────────────────────────────────
    # Processes historical bricks' coarse representation sequentially
    c = TimeDistributed(
        Conv1D(16, kernel_size=1, activation='relu', padding='causal')
    )(history_input)  # Shape: (batch, 5, 5, 16)
    c = TimeDistributed(Flatten())(c)    # Shape: (batch, 5, 80)
    c = LSTM(16, return_sequences=False, kernel_regularizer=l2(l2_reg))(c)
    c = Dropout(0.2)(c)
    
    # ── Fusion ───────────────────────────────────────────────────
    fused = Concatenate()([a, b, c])              # Shape: (64,)
    fused = Concatenate()([fused, scalar_input])  # Shape: (64 + 34 = 98,)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(fused)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.2)(x)
    
    # ── Outputs ──────────────────────────────────────────────────
    # Primary setup probability head (5-class softmax)
    setup_probs = Dense(5, activation='softmax', name='setup_probs')(x)
    
    # Auxiliary binary heads for T1, T2, T3, T4 outcomes to combat class starvation
    aux_t1 = Dense(1, activation='sigmoid', name='aux_t1')(x)
    aux_t2 = Dense(1, activation='sigmoid', name='aux_t2')(x)
    aux_t3 = Dense(1, activation='sigmoid', name='aux_t3')(x)
    aux_t4 = Dense(1, activation='sigmoid', name='aux_t4')(x)
    
    model = Model(
        inputs=[ancs_fine_input, ancs_coarse_input, history_input, scalar_input],
        outputs=[setup_probs, aux_t1, aux_t2, aux_t3, aux_t4],
        name='setup_classifier_training'
    )
    
    return model

def get_inference_model(training_model: Model) -> Model:
    """
    Extracts the inference-only sub-model by keeping only the primary softmax output.
    """
    # Create a new model with the same inputs, mapping directly to the 'setup_probs' output layer.
    setup_probs_output = training_model.get_layer('setup_probs').output
    inference_model = Model(
        inputs=training_model.inputs,
        outputs=setup_probs_output,
        name='setup_classifier_inference'
    )
    return inference_model
