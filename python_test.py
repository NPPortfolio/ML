import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import pathlib

#  Tensorflow tutorial "Load and preprocess images"

train_folder = pathlib.Path('./data/train')
val_folder = pathlib.Path('./data/val')

batch_size = 32
img_height = 256
img_width = 256

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_folder,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_folder,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

print(train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE # Tune the buffer size dynamically at runtime

# cache() keeps images in memory after first epoch
# prefetch() overlaps image preprocessing and model execution (total time is max instead of sum)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Tensorflow tutorial "Image classification"

# model = tf.keras.Sequential([
  # tf.keras.layers.experimental.preprocessing.Rescaling(1./255), # Convert rbg values from [0, 255] to [0, 1]
# ])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))

model.summary()


