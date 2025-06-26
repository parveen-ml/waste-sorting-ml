import tensorflow as tf
from tensorflow.keras import layers, models
from data_loader import get_datasets

# Load datasets
train_ds, val_ds, test_ds = get_datasets("data/train", "data/val", "data/test")

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# MobileNetV2 base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze base for transfer learning

# Add classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
for batch, labels in train_ds.take(1):
    for i in range(len(batch)):
        try:
            _ = batch[i].numpy()  # Force image load
        except Exception as e:
            print(f"❌ Failed to load image in batch: {i}")
            print(f"Error: {e}")
# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  
)

# Save model
model.save("models/mobilenet_waste_classifier.h5")
