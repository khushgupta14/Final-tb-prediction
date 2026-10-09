import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "tb_model.h5")

# ── Settings ─────────────────────────────────────────────
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ── Load all image paths and labels ──────────────────────
def load_paths_and_labels():
    paths = []
    labels = []
    normal_dir = os.path.join(DATASET_DIR, "Normal")
    for fname in os.listdir(normal_dir):
        paths.append(os.path.join(normal_dir, fname))
        labels.append(0)
    tb_dir = os.path.join(DATASET_DIR, "TB")
    for fname in os.listdir(tb_dir):
        paths.append(os.path.join(tb_dir, fname))
        labels.append(1)
    return paths, labels

paths, labels = load_paths_and_labels()

# ── Shuffle ──────────────────────────────────────────────
combined = list(zip(paths, labels))
random.shuffle(combined)
paths, labels = zip(*combined)
paths = list(paths)
labels = list(labels)

total = len(paths)
print(f"Total images: {total}")

# ── 70 / 15 / 15 split ──────────────────────────────────
train_end = int(total * 0.70)
val_end = int(total * 0.85)

train_paths, train_labels = paths[:train_end], labels[:train_end]
val_paths,   val_labels   = paths[train_end:val_end], labels[train_end:val_end]
test_paths,  test_labels  = paths[val_end:], labels[val_end:]

print(f"Training:   {len(train_paths)} images")
print(f"Validation: {len(val_paths)} images")
print(f"Testing:    {len(test_paths)} images")

# ── Load images into numpy arrays ────────────────────────
def load_images(file_paths):
    images = []
    for i, p in enumerate(file_paths):
        img = image.load_img(p, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        images.append(img_array)
        if (i + 1) % 500 == 0:
            print(f"  Loaded {i + 1}/{len(file_paths)}")
    return np.array(images)

print("\nLoading training images...")
X_train = load_images(train_paths)
y_train = np.array(train_labels)

print("Loading validation images...")
X_val = load_images(val_paths)
y_val = np.array(val_labels)

print("Loading test images...")
X_test = load_images(test_paths)
y_test = np.array(test_labels)

# ── Data augmentation (mild, on training data only) ──────
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
)
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# ── Gentle class weights ─────────────────────────────────
class_weights = {0: 1.0, 1: 2.0}
print(f"\nClass weights: Normal=1.0, TB=2.0")

# ── CNN model ────────────────────────────────────────────
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Early stopping ───────────────────────────────────────
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

# ── Train ────────────────────────────────────────────────
steps_per_epoch = len(X_train) // BATCH_SIZE

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop],
)

# ── Test evaluation ──────────────────────────────────────
print("\n── Test Set Evaluation ──")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss:     {test_loss:.4f}")

# ── Save ─────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")