import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

# ── 1. Load Model ────────────────────────────────────────
MODEL_PATH = "tb_model.h5"
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found!")
    exit()

print(f"Loading {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# ── 2. Load Test Dataset ─────────────────────────────────
print("Loading test images from dataset/test...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Rescale pixels to [0, 1]
rescale = layers.Rescaling(1.0 / 255)
test_ds_rescaled = test_ds.map(lambda x, y: (rescale(x), y))

# ── 3. Quick Built-in Accuracy ───────────────────────────
loss, accuracy = model.evaluate(test_ds_rescaled, verbose=0)

# ── 4. Detailed Calculation (TP, TN, FP, FN) ─────────────
y_true = []
y_pred = []

for images, labels in test_ds_rescaled:
    preds = model.predict(images, verbose=0).flatten()
    y_true.extend(labels.numpy())
    y_pred.extend((preds >= 0.5).astype(int))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion Matrix components
# Class 0: Normal, Class 1: Tuberculosis
TP = int(np.sum((y_pred == 1) & (y_true == 1)))
TN = int(np.sum((y_pred == 0) & (y_true == 0)))
FP = int(np.sum((y_pred == 1) & (y_true == 0)))
FN = int(np.sum((y_pred == 0) & (y_true == 1)))

total_samples = len(y_true)
calc_acc = (TP + TN) / total_samples
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0   # Recall for TB
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0   # For Normal

print("\n" + "=" * 50)
print("          MODEL ACCURACY & EVALUATION REPORT       ")
print("=" * 50)
print(f"Total Test Images Tested: {total_samples}")
print(f"Overall Accuracy:         {calc_acc * 100:.2f}%")
print(f"Sensitivity (Recall):     {sensitivity * 100:.2f}% (TB Detection Rate)")
print(f"Specificity:              {specificity * 100:.2f}% (Normal Clearance Rate)")
print("-" * 50)
print("Confusion Matrix:")
print(f"  True Positives  (TB caught):         {TP}")
print(f"  True Negatives  (Normal caught):     {TN}")
print(f"  False Positives (Normal marked TB):  {FP}")
print(f"  False Negatives (TB missed):         {FN}")
print("=" * 50)
