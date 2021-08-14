import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

def create_discriminator():

    # Test dropout a bit

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[256, 256, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

def create_generator():

    # Have to figure out what exactly these parameters do and the best structure for the network, sort of messing around right now

    # Conv2DTranspose upsamples and performs a convolution, sort of like a "learned" upsample
    # A stride over 1 adds gaps to the input image, which does the upsampling
    # From a tutorial: "It is also good practice to use a kernel size that is a factor of the stride (e.g. double) to avoid a checkerboard pattern
    # that can sometimes be observed when upsampling.""

    # alpha at 0.2 best practice? more info

    # NOTE: I think this input shape is where each label has to have some influence
    model = tf.keras.Sequential()
    model.add(layers.Dense(64 * 32 * 32, input_shape=(100,), activation='relu'))
    model.add(layers.BatchNormalization()) # Transformation that maintains the mean output close to 0 and SD close to 1
    model.add(layers.LeakyReLU(alpha=0.2)) # Question whether to have this before or after activation function

    model.add(layers.Reshape((32, 32, 64))) # Question about rbg channels here
    print(model.output.shape)

    model.add(layers.Conv2DTranspose(32, (8, 8), strides=(2,2), padding='same', use_bias=False)) # 32 64 x 64 filters
    print(model.output.shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(16, (12, 12), strides=(2,2), padding='same', use_bias=False)) # 16 128 * 128 filters
    print(model.output.shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Important note: probably want more than one filter here, next layer defines final image
    model.add(layers.Conv2DTranspose(16, (16, 16), strides=(2,2), padding='same', use_bias=False)) # 16 256 * 256 filters
    print(model.output.shape)

    # This output layer has 3 256 * 256 filters (rgb), tanh makes values between -1 and 1 
    model.add(layers.Conv2D(3, (3,3), activation='tanh', padding='same'))
    print(model.output.shape)

    return model

generator = create_generator()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()











# tutorial source for later
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

# Probably not going to use

def create_gan(dis, gen):
    # make weights in the discriminator not trainable
    # From the tutorial, this means that the standalone discriminator can be trained but the wieghts will not update when GAN is trained?
	dis.trainable = False
	# connect them
	model = tf.keras.layers.Sequential()
	# add generator
	model.add(gen)
	# add the discriminator
	model.add(dis)
	# compile model
	opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model