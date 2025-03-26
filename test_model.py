# test_model.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# === CONFIG ===
IMAGE_SIZE = 128
MODEL_PATH = "analog_clocks/saved_model/clock_regression_model.h5"
IMAGE_DIR = "analog_clocks/images"

# === LOAD MODEL ===
model = tf.keras.models.load_model(MODEL_PATH)

# === PREDICT ON NEW IMAGE ===
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img_input = np.expand_dims(img, axis=0)
    pred = model.predict(img_input)[0]
    plt.imshow(img)
    plt.title(f"Predicted: {pred[0]:.1f}h {pred[1]:.1f}m")
    plt.axis('off')
    plt.show()

# === EXAMPLE USAGE ===
# Replace this with the actual image filename you want to test
example_file = os.path.join(IMAGE_DIR, "48888.jpg")
predict_image(example_file)
