# train_model.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import matplotlib.pyplot as plt

# === CONFIG ===
IMAGE_SIZE = 128
DATA_DIR = "analog_clocks/processed"
MODEL_DIR = "analog_clocks/saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# === LOAD PROCESSED DATA ===
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

# === NORMALIZE LABELS ===
y[:, 0] /= 11.0   # hour (0–11)
y[:, 1] /= 59.0   # minute (0–59)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# === MODEL DEFINITION ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='linear')  # [hour, minute] regression output
])

model.compile(
    optimizer=Adam(1e-4),
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError()]
)

# === TRAINING ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# === SAVE MODEL ===
model.save(os.path.join(MODEL_DIR, "clock_regression_model.h5"))
print(f"Model saved to {os.path.join(MODEL_DIR, 'clock_regression_model.h5')}")

# === EVALUATION ===
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {mae:.2f}")

# === PREDICTION VISUALIZATION ===
def plot_prediction(idx):
    pred = model.predict(X_test[idx:idx+1])[0]
    pred[0] *= 11.0
    pred[1] *= 59.0
    actual = y_test[idx].copy()
    actual[0] *= 11.0
    actual[1] *= 59.0
    plt.imshow(X_test[idx])
    plt.title(f"Predicted: {pred[0]:.1f}h {pred[1]:.1f}m\nActual: {actual[0]:.1f}h {actual[1]:.1f}m")
    plt.axis('off')
    plt.show()

# Example usage: 
plot_prediction(0)
plot_prediction(1)
plot_prediction(2)
