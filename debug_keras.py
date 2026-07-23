import sys, os, time
from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

print("Generating dummy data...")
# smaller data to be safe
micro = np.random.rand(100, 10, 100, 9).astype(np.float32)
macro = np.random.rand(100, 10, 3).astype(np.float32)
y_class = np.random.randint(0, 2, size=(100, 1)).astype(np.float32)
y_mag = np.random.rand(100, 1).astype(np.float32)

from tensorflow.keras.utils import Sequence
class DataGenerator(Sequence):
    def __init__(self, micro, macro, y_class, y_mag, batch_size=64, **kwargs):
        super().__init__(**kwargs)
        self.micro = micro
        self.macro = macro
        self.y_class = y_class
        self.y_mag = y_mag
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.y_class) / self.batch_size))

    def __getitem__(self, idx):
        print(f"Generator called for batch {idx}!")
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.y_class))
        batch_x = (self.micro[start:end], self.macro[start:end])
        batch_y = (self.y_class[start:end], self.y_mag[start:end])
        return batch_x, batch_y

train_gen = DataGenerator(micro, macro, y_class, y_mag)
val_gen = DataGenerator(micro, macro, y_class, y_mag)

print("Importing models_exec...")
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
from models_exec import build_baseline_exec_model, compile_model

print("Building model...")
model = build_baseline_exec_model()
model = compile_model(model)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1, min_lr=1e-6),
    ModelCheckpoint(filepath="outputs/exec_baseline/model.keras", monitor='val_loss', save_best_only=True, verbose=1),
    CSVLogger("outputs/exec_baseline/training_log.csv"),
]

print("Calling fit...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2,
    callbacks=callbacks,
    verbose=1
)
print("Finished!")
