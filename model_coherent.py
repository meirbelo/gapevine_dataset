import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from keras.optimizers import Adam

# Data augmentation pour l'ensemble d'entraînement
train_datagen = ImageDataGenerator(
    rescale=1./255
)

# Créer le générateur pour l'ensemble d'entraînement
train_generator = train_datagen.flow_from_directory(
    directory="content/repartition/train",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True
)

# Data augmentation pour l'ensemble de validation
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    directory="content/repartition/validation",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True
)

# Data augmentation pour l'ensemble de test
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory="content/repartition/test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Charger le modèle VGG16 pré-entraîné sur ImageNet et sans la dernière couche de classification
base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))

# Ajouter une nouvelle couche de classification
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Définir le modèle final
model = Model(inputs=base_model.input, outputs=predictions)

# Congeler les couches du modèle de base pour ne pas les entraîner
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001, decay=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Entraîner le modèle avec les générateurs d'images
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Évaluer le modèle sur l'ensemble de test
scores = model.evaluate(
    test_generator,
    steps=test_generator.samples // test_generator.batch_size
)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
