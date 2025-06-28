# Podejscie 2 : transfer learning z MobileNetV2, uzycie gotowego modelu trenowanego na waielkim zbiorze ImageNet
# nie budowany od 0, doklejamy tylko ostatnie warstwy klasyfikujace do moich klas

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_dir = os.path.join(base_dir, 'data', 'train')
val_dir = os.path.join(base_dir, 'data', 'val')

# Parametry
img_size = (224, 224) #rozmiar zdjec, siec neuronowa MobileNetV2 wymaga min 96x96 a najlepiej 224x224
batch_size = 16 #liczba probek w batchu(male porcje zdjec)

# Generatory danych z argumentacja
train_datagen = ImageDataGenerator(
    rescale=1./255, #normalizacja pikseli 0-225 -> 0-1
    rotation_range=20, # losowy obrot
    width_shift_range=0.1, #przesuniecie w poziomie
    height_shift_range=0.1, #przesuniecie w pionie
    zoom_range=0.1, #zoom
    horizontal_flip=True #odbicie lustrzane
)
#Generator walidacyjny (tylko skalowanie)
val_datagen = ImageDataGenerator(rescale=1./255)
#Wczytywanie danych treningowych
train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
#Wczytywanie danych walidacyjnych
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

# Wczytywanie baowego modelu MobileNetV2 bez górnych warstw klasyfikacyjnych
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # zamrażamy warstwy

# Doklejamy własny „head” klasyfikator
model = models.Sequential([
    base_model, # baza z MobileNeta
    layers.GlobalAveragePooling2D(), # zamienia ostatnią macierz cech na wektor
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),# drop dla lepszej generalizacji
    layers.Dense(3, activation='softmax') #warstwa wyjściowa na 3 klasy
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint_cb = ModelCheckpoint('mobilenet_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
early_stop_cb = EarlyStopping(patience=4, restore_best_weights=True, monitor='val_loss')

# Trening modelu
history = model.fit(
    train_gen,
    epochs=25,
    validation_data=val_gen,
    callbacks=[checkpoint_cb, early_stop_cb]
)

# Wykresy
plt.figure(figsize=(12,5))

#Dokladnosc(accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność')
plt.legend()
#Strata (loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Strata')
plt.legend()

plt.tight_layout()
plt.show()
