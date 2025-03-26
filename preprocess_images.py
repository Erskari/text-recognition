import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# === CONFIG ===
IMAGE_SIZE = 128
LIMIT = 10000
DATA_DIR = "analog_clocks/"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
LABEL_FILE = os.path.join(DATA_DIR, "label.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD LABELS ===
labels_df = pd.read_csv(LABEL_FILE)
image_files = sorted(os.listdir(IMAGE_DIR), key=lambda x: int(x.split('.')[0]))[:LIMIT]

# === LOAD & PREPROCESS IMAGES ===
images = []
labels = []

for idx, file in enumerate(tqdm(image_files, desc='Processing images')):
    img_path = os.path.join(IMAGE_DIR, file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    images.append(img)
    labels.append(labels_df.iloc[idx].values)

X = np.array(images, dtype=np.float32)
y = np.array(labels, dtype=np.float32)

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
print(f"Saved {len(X)} samples to {OUTPUT_DIR}/")
