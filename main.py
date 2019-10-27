from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import load_data

train_labels, train_images = load_data.loadData("bin/labels.csv", "bin/images")

print("DATA")

# Ustawienie layerów modelu
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64, 64)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# Ustawienie przed kompilowaniem
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("GO COMPILE!")
# Kompilacja modelu
train_labels = np.argmax(train_labels, axis=1)
model.fit(
    x=train_images, 
    y=train_labels, 
    batch_size=None, 
    epochs=256,
    verbose=1,
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_freq=1,
    max_queue_size=10,
    workers=5,
    use_multiprocessing=True
    )

print("EVALUATE!")
test_loss, test_acc = model.evaluate(train_images,  train_labels, verbose=0)
print('\nTest accuracy:', test_acc, '\nTest loss:', test_loss)

model.save("dest/model.h5")

del model