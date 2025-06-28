# Podejscie 1 : trening od 0 Custom CNN - koniecznosc dlugiego uczenia na duzych zbiorach dancyh


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

train_dir = '../data/train'
val_dir = '../data/val'

# Augmentacja + normalizacja danych treningowych
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Walidacja tylko skalowana
val_datagen = ImageDataGenerator(rescale=1./255)

# Generator danych
train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=16, class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=16, class_mode='categorical'
)

# Budowa modelu z nowym Input
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks: zapis najlepszego modelu + wczesne zatrzymanie
checkpoint_cb = ModelCheckpoint('najlepszy_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
early_stop_cb = EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss')

# Trenowanie modelu
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=[checkpoint_cb, early_stop_cb]
)

# Wykresy dokładności i straty
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Strata')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

plt.tight_layout()
plt.show()
