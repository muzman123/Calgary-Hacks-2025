#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# =====================================================
# ========== 1) Configuration and Setup  ==============
# =====================================================
spectrogram_dir = '/Users/hamzakhurram/Documents/ML_Model/scripts/spectrograms'
img_height, img_width = 128, 128
batch_size = 32
epochs = 40

# =====================================================
# ========== 2) Data Generators with Augmentation =====
# =====================================================
# Data augmentation to reduce overfitting:
#   - small random rotations
#   - random zoom
#   - horizontal flips (if your spectrogram orientation can be flipped)
#   - etc.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.3,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=False,  # Often not beneficial for spectrograms, but can be True for generic images
    width_shift_range=0.1,
    height_shift_range=0.1,
)

# Validation generator uses only rescaling, no augmentation
valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.3
)

# Training set (70% of data)
train_generator = train_datagen.flow_from_directory(
    spectrogram_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation set (30% of data)
validation_generator = valid_datagen.flow_from_directory(
    spectrogram_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Print and save the class indices
print("Class indices mapping:", train_generator.class_indices)
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# =====================================================
# ========== 3) Build Model with Dropout Layers =======
# =====================================================
# A simple CNN, but with dropout to reduce overfitting
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),  # Dropout #1

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),  # Dropout #2

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),  # Dropout #3

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Dropout #4
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================================================
# ========== 4) Define Callbacks (Early Stop, etc.) ===
# =====================================================
# EarlyStopping: stop if val_accuracy doesn't improve for 'patience' epochs
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,    # stop after 10 epochs of no improvement
    restore_best_weights=True
)

# ModelCheckpoint: save best model during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# =====================================================
# ========== 5) Train Model ===========================
# =====================================================
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, checkpoint],
)

# If early stopping triggered, 'best_model.h5' is the best checkpoint
print("Training complete. Loading best model weights from checkpoint if needed.")
model.load_weights('best_model.h5')

# Save the final model under a different filename
model.save('sound_classification_model.h5')
print("Final model saved as sound_classification_model.h5")