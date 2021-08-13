import tensorflow as tf
import matplotlib.pyplot as plt

# This is where I'll start on the generator part of the gan, a lot to look into

# Goal is to go from noise to a 256 * 256 * 3 image


def create_generator():

    # Have to figure out what exactly these parameters do and the best structure for the network, sort of messing around right now

    # Conv2DTranspose upsamples and performs a convolution, sort of like a "learned" upsample
    # A stride over 1 adds gaps to the input image, which does the upsampling
    # From a tutorial: "It is also good practice to use a kernel size that is a factor of the stride (e.g. double) to avoid a checkerboard pattern
    # that can sometimes be observed when upsampling.""

    # alpha at 0.2 best practice? more info

    # NOTE: I think this input shape is where each label has to have some influence
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64 * 32 * 32, input_shape=(100,), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization()) # Transformation that maintains the mean output close to 0 and SD close to 1
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2)) # Question whether to have this before or after activation function

    model.add(tf.keras.layers.Reshape((32, 32, 64))) # Question about rbg channels here
    print(model.output.shape)

    model.add(tf.keras.layers.Conv2DTranspose(32, (8, 8), strides=(2,2), padding='same', use_bias=False))
    print(model.output.shape)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2DTranspose(16, (12, 12), strides=(2,2), padding='same', use_bias=False))
    print(model.output.shape)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # Important note: probably want more than one filter here, next layer defines final image
    model.add(tf.keras.layers.Conv2DTranspose(16, (16, 16), strides=(2,2), padding='same', use_bias=False))
    print(model.output.shape)

    # This output layer has 3 256 * 256 filters (rgb), tanh makes values between -1 and 1 
    model.add(tf.keras.layers.Conv2D(3, (3,3), activation='tanh', padding='same'))
    #assert model.output_shape == (None, 256, 256, 3)  # None is batch size?

    return model

generator = create_generator()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()
