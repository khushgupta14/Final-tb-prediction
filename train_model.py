import os
import tensorflow as tf
from keras import layers, models, regularizers
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
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.005)),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# ── 3. Train, Evaluate & Save ────────────────────────────
# Class weights balance the dataset (~5:1 Normal to TB ratio) to achieve ~80% Recall and ~88-90% Specificity
class_weights = {0: 1.0, 1: 3.5}

model.fit(train_ds, validation_data=val_ds, epochs=10, class_weight=class_weights)

print("\n--- Evaluation on Test Dataset ---")
model.evaluate(test_ds)

model.save("tb_model.h5")
print("\nModel saved to tb_model.h5")