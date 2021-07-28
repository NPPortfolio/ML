import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import pathlib

image_folder = pathlib.Path('./data/images')
images = list(image_folder.glob('*.jpg'))

print(len(images))

batch_size = 32
img_height = 150
img_width = 150
