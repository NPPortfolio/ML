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

train_ds = train_ds.take(10)
val_ds = val_ds.take(10)

# Tune the buffer size dynamically at runtime
AUTOTUNE = tf.data.AUTOTUNE

# cache() keeps images in memory after first epoch
# prefetch() overlaps image preprocessing and model execution (total time is max instead of sum)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 8

# 1st param (16, 32, 64) Dimensionality of output space, or number of output filters in convolution
# 2nd param (5): Kernel size, 5x5 matrix of weights to slide over the input data

# MaxPooling() : Downsamples the input, by default it is by half in each dimension (1/4)

# Dense() : Layer where every node in the previous layer connects to every node in this (dense) layer

# relu : Max between 0 and input

model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# Adam is a variant of SGD

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.summary()

# Training

epochs = 5

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



