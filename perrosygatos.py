import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt

# Descargar el set de datos de perros y gatos
train_dir = "G:\Trabajo\Curso Programación IA\IA\python\kagglecatsanddogs_5340\PetImages"
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalización de las imágenes
    rotation_range=40,  # Rotación aleatoria
    width_shift_range=0.2,  # Desplazamiento horizontal aleatorio
    height_shift_range=0.2,  # Desplazamiento vertical aleatorio
    shear_range=0.2,  # Transformación de cizalla aleatoria
    zoom_range=0.2,  # Zoom aleatorio
    horizontal_flip=True,  # Volteo horizontal aleatorio
    fill_mode='nearest'  # Relleno de los píxeles que faltan
)

# Crear imágenes de prueba
test_datagen = ImageDataGenerator(rescale=1./255)

# Flujo de datos de entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Redimensionar las imágenes a 150x150 píxeles
    batch_size=32,  # Tamaño del lote
    class_mode='binary'  # Como es clasificación binaria (gato vs perro)
)

# ********** MODELO CNN **************
# Definir el modelo secuencial
model = models.Sequential([
    # Primera capa convolucional
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    # Segunda capa convolucional
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Capa de aplanado
    layers.Flatten(),

    # Capa densa (fully connected)
    layers.Dense(128, activation='relu'),

    # Capa de salida
    layers.Dense(1, activation='sigmoid')  # Salida binaria: gato o perro
])

# Compilar el modelo
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Número de pasos por época
    epochs=10,  # Número de épocas
)

# VALIDAR RENDIMIENTO
# Graficar la precisión y la pérdida
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
# plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
# plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend(loc='upper right')
plt.show()
