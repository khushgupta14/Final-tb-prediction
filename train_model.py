import os
import tensorflow as tf
from keras import layers, models
from keras.applications import VGG16

# ── 1. Load Train, Val, and Test directly from folders ───
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train", image_size=(224, 224), batch_size=32
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/val", image_size=(224, 224), batch_size=32
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test", image_size=(224, 224), batch_size=32
)

# Normalize pixels to [0, 1]
rescale = layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (rescale(x), y))
val_ds = val_ds.map(lambda x, y: (rescale(x), y))
test_ds = test_ds.map(lambda x, y: (rescale(x), y))

# ── 2. VGG16 Transfer Learning Model ─────────────────────
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze VGG16

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ── 3. Train, Evaluate & Save ────────────────────────────
model.fit(train_ds, validation_data=val_ds, epochs=10)

print("\n--- Evaluation on Test Dataset ---")
model.evaluate(test_ds)

model.save("tb_model.h5")
print("\nModel saved to tb_model.h5")