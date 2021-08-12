import tensorflow as tf
import matplotlib.pyplot as plt

# This is where I'll start on the generator part of the gan, a lot to look into

# Goal is to go from noise to a 256 * 256 * 3 image


def create_generator():

    # Have to figure out what exactly these parameters do and the best structure for the network, sort of messing around right now

    # NOTE: I think this input shape is where each label has to have some influence
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64 * 4 * 4, input_shape=(100,), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization()) # Transformation that maintains the mean output close to 0 and SD close to 1
    model.add(tf.keras.layers.LeakyReLU()) # Question whether to have this before or after activation function
    print(model.output.shape)

    model.add(tf.keras.layers.Reshape((4, 4, 64))) # Question about rbg channels here
    print(model.output.shape)

    # Conv2DTranspose upsamples and performs a convolution
   # model.add(tf.keras.layers.Conv2DTranspose(32, (1, 1), strides=(2,2), padding='same', use_bias=False))
   # print(model.output.shape)
   # model.add(tf.keras.layers.BatchNormalization())
   # model.add(tf.keras.layers.LeakyReLU())

    #model.add(tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=(4,4), padding='same', use_bias=False))
    #print(model.output.shape)
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (1, 1), strides=(8,8), padding='same', use_bias=False))
    print(model.output.shape)

    #assert model.output_shape == (None, 256, 256, 3)  # None is batch size?

    return model

generator = create_generator()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()
