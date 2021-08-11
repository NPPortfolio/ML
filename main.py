import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import pathlib

# Note: Right now I'm just running these functions from vscodes terminal and the python interpreter so I can train, save, and load the model easier
# There could be something wrong with doing 1 epoch at a time though, fit might scramble the inputs between a multi epoch call, need to look into more
# Also need to look into the datasets caching, see if this setup messes it up

def create_datasets():

    train_folder = pathlib.Path('./data/train')
    test_folder = pathlib.Path('./data/test')

    batch_size = 32
    img_height = 256
    img_width = 256

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_folder,
        seed=123,
        validation_split=0.1,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Allows you to see how well the model is generalizing during training (not overfitting)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_folder,
        seed=123,
        validation_split=0.1,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Test the trained model with these, shouldn't have labels
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_folder,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    train_ds = train_ds.take(100)
    val_ds = val_ds.take(10)

    # Tune the buffer size dynamically at runtime
    AUTOTUNE = tf.data.AUTOTUNE

    # cache() keeps images in memory after first epoch
    # prefetch() overlaps image preprocessing and model execution (total time is max instead of sum)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return (train_ds, val_ds, test_ds)

def create_model():

    num_classes = 8

    batch_size = 32
    img_height = 256
    img_width = 256

    # 1st param (16, 32, 64) Dimensionality of output space, or number of output filters in convolution
    # 2nd param (5): Filter size, 5x5 matrix of weights to slide over the input data
    # 3rd param (padding): 'Same' pads input with 0's so the output has the same dimensions as the input
    # 4th param (activation): Activation function, 'relu' is maximum of 0 and input

    # MaxPooling() : Downsamples the next layer's input (new representation of input created by previous layer's convolution),
    #                by default it is by half in each dimension (1/4). MaxPooling takes maximum value of 2x2 square of pixels
    # ^ Picks out the "most activated" pixels and preserves the values moving forward

    # Dense() : Layer where every node in the previous layer connects to every node in this (dense) layer

    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(
            1./255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        #tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
        # tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        # Need to figure out what to do for these next two, right now I think this is wrong dimension wise
        # tf.keras.layers.Dense(64, activation='relu'),
        # The final layer that tells how activated each label is
        # TODO: want something like softmax for last layer
        tf.keras.layers.Dense(num_classes)
    ])

    # Adam is a variant of SGD

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])

    model.summary()

    return model

# Mutator, need to know for interpreter
def train_model(model, train_ds, val_ds, epochs):

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )

# hardcoded but dont want to keep passing in filename
def load_model():
    model = 0
    try:
        model = tf.keras.models.load_model('model.h5')
    except (ImportError, IOError) as e:
        print(e)
        return None # bad
    return model

def save_model(model):
    model.save('model.h5')

# testing
d = create_datasets()