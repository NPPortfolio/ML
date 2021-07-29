import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import pathlib

# image_folder = pathlib.Path('./data/images')
# images = list(image_folder.glob('*.jpg'))

list_ds = tf.data.Dataset.list_files('./data/images/*.jpg', shuffle=False)

image_count = len(list_ds)

# Tensorflow tutorial functions
def img_from_path(file_path):
    img = tf.io.read_file(file_path)
    return decode_img(img)

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    return tf.io.decode_jpeg(img, channels=3)

for e in list_ds.take(1):
    print(img_from_path(e))


