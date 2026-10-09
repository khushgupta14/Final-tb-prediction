import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "tb_model.h5")
IMG_SIZE = (150, 150)

# Load paths and labels
paths, labels = [], []
for fname in os.listdir(os.path.join(DATASET_DIR, "Normal")):
    paths.append(os.path.join(DATASET_DIR, "Normal", fname))
    labels.append(0)
for fname in os.listdir(os.path.join(DATASET_DIR, "TB")):
    paths.append(os.path.join(DATASET_DIR, "TB", fname))
    labels.append(1)

paths = np.array(paths)
labels = np.array(labels)

# Stratified split: keeps exact ratio of Normal and TB in the 15% test set
_, test_paths, _, y_test = train_test_split(
    paths, labels, test_size=0.15, random_state=42, stratify=labels
)

print(f"Test Set Count: Normal={(y_test==0).sum()}, TB={(y_test==1).sum()}")

# Load test images
X_test = np.array([
    image.img_to_array(image.load_img(p, target_size=IMG_SIZE)) / 255.0 
    for p in test_paths
])

# Predictions
model = load_model(MODEL_PATH)
y_pred_prob = model.predict(X_test, batch_size=32).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

# Classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "TB"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Standard ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig(os.path.join(BASE_DIR, "roc_curve_standard.png"), dpi=300)
plt.close()

print(f"\nStandard AUC: {roc_auc:.4f}")
print("Saved image: roc_curve_standard.png")