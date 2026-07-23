import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from sc_model import build_setup_classifier, get_inference_model

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Trainer")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "setup_classifier"
TENSOR_DIR = OUTPUT_DIR / "tensors"

def prepare_targets(data_dict, smooth=True):
    # Parse inputs
    x = [
        data_dict['ancs_fine'],
        data_dict['ancs_coarse'],
        data_dict['history'],
        data_dict['scalars']
    ]
    
    # 5-class target preparation
    labels = data_dict['label']
    one_hot = np.eye(5)[labels]
    
    if smooth and 'boundary_proximity' in data_dict.files:
        # Conditional Label Smoothing based on boundary proximity
        boundary_prox = data_dict['boundary_proximity']
        epsilon = 0.05 * np.exp(-10.0 * boundary_prox)
        # Apply smoothing: (1 - eps) * one_hot + eps / 5
        smoothed_labels = (1.0 - epsilon)[:, None] * one_hot + (epsilon / 5.0)[:, None]
        # Cast to float32
        smoothed_labels = smoothed_labels.astype(np.float32)
    else:
        smoothed_labels = one_hot.astype(np.float32)
        
    # Order matches model.output_names: ['setup_probs', 'aux_t1', 'aux_t2', 'aux_t3', 'aux_t4']
    y = [
        smoothed_labels,
        data_dict['t1_win'].astype(np.float32),
        data_dict['t2_win'].astype(np.float32),
        data_dict['t3_win'].astype(np.float32),
        data_dict['t4_win'].astype(np.float32),
    ]
    
    return x, y, labels

def main():
    # 1. Load splits
    train_path = TENSOR_DIR / "train.npz"
    val_path = TENSOR_DIR / "val.npz"
    
    if not train_path.exists() or not val_path.exists():
        logger.error("Tensor files not found. Run sc_tensor_builder.py first.")
        return
        
    logger.info("Loading train and validation tensors...")
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    
    # 2. Prepare features and targets
    x_train, y_train, train_labels = prepare_targets(train_data, smooth=True)
    x_val, y_val, _ = prepare_targets(val_data, smooth=False) # No smoothing on validation targets
    
    logger.info(f"Training set size: {x_train[0].shape[0]} samples")
    logger.info(f"Validation set size: {x_val[0].shape[0]} samples")
    
    # 3. Load class weights for sample weighting the primary loss
    weights_path = OUTPUT_DIR / "class_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            class_weights_dict = json.load(f)
        logger.info(f"Loaded class weights: {class_weights_dict}")
        # Map class weights to each sample in the training set
        train_sample_weights = np.array([class_weights_dict[str(lbl)] for lbl in train_labels], dtype=np.float32)
    else:
        logger.warning("class_weights.json not found. Training without sample weights.")
        train_sample_weights = np.ones(len(train_labels), dtype=np.float32)
        
    # Map to all outputs since Keras requires one sample_weight array per output in y
    N_train = len(train_labels)
    sample_weights = [
        train_sample_weights,
        np.ones(N_train, dtype=np.float32),
        np.ones(N_train, dtype=np.float32),
        np.ones(N_train, dtype=np.float32),
        np.ones(N_train, dtype=np.float32)
    ]
        
    # 4. Build and Compile Model
    logger.info("Building multi-branch SetupClassifier model...")
    model = build_setup_classifier()
    
    losses = [
        'categorical_crossentropy', # setup_probs
        'binary_crossentropy',       # aux_t1
        'binary_crossentropy',       # aux_t2
        'binary_crossentropy',       # aux_t3
        'binary_crossentropy'        # aux_t4
    ]
    
    loss_weights = [
        0.40, # setup_probs
        0.15, # aux_t1
        0.15, # aux_t2
        0.15, # aux_t3
        0.15  # aux_t4
    ]
    
    # List of metrics corresponding to each output index
    metrics = [
        ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')], # setup_probs
        [], # aux_t1
        [], # aux_t2
        [], # aux_t3
        []  # aux_t4
    ]
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    # Print summary
    model.summary()
    
    # 5. Set up callbacks
    model_path = OUTPUT_DIR / "model.keras"
    logger_path = OUTPUT_DIR / "training_log.csv"
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(model_path), monitor='val_loss', save_best_only=True, verbose=1),
        CSVLogger(str(logger_path))
    ]
    
    # 6. Fit Model
    BATCH_SIZE = 64
    MAX_EPOCHS = 150
    
    logger.info("Starting model training...")
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=sample_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Save prediction-only inference model (without auxiliary outputs)
    logger.info("Training complete. Building optimized inference model...")
    if model_path.exists():
        model.load_weights(str(model_path))
        
    inference_model = get_inference_model(model)
    inference_model_path = OUTPUT_DIR / "model_inference.keras"
    inference_model.save(str(inference_model_path))
    logger.info(f"Optimized inference model successfully saved to {inference_model_path}")

if __name__ == "__main__":
    main()
