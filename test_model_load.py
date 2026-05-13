import tensorflow as tf
from src.model import build_model, compile_model

model_path = "BrickOfTicks_Trader/models/fold_1/model.keras"
model = tf.keras.models.load_model(model_path, compile=False)
model = compile_model(model)
print("Model loaded successfully")
