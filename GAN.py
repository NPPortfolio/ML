import tensorflow as tf
import matplotlib.pyplot as plt
import time

from tensorflow.keras import layers

# https://www.tensorflow.org/tutorials/generative/dcgan
# Right now this file is mostly the code from the above tutorial
# As I go I will tweak to fit with the landscape generator ideas


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
    # Transformation that maintains the mean output close to 0 and SD close to 1
    model.add(layers.BatchNormalization())
    # Question whether to have this before or after activation function
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((32, 32, 64)))  # Question about rbg channels here
    print(model.output.shape)

    model.add(layers.Conv2DTranspose(32, (8, 8), strides=(2, 2),
              padding='same', use_bias=False))  # 32 64 x 64 filters
    print(model.output.shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(16, (12, 12), strides=(2, 2),
              padding='same', use_bias=False))  # 16 128 * 128 filters
    print(model.output.shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Important note: probably want more than one filter here, next layer defines final image
    model.add(layers.Conv2DTranspose(16, (16, 16), strides=(2, 2),
              padding='same', use_bias=False))  # 16 256 * 256 filters
    print(model.output.shape)

    # This output layer has 3 256 * 256 filters (rgb), tanh makes values between -1 and 1
    model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    print(model.output.shape)

    return model


generator = create_generator()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()

discriminator = create_discriminator()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Each network trained separately, different optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

BATCH_SIZE = 1
EPOCHS = 1
noise_dim = 100
num_examples_to_generate = 16


@tf.function  # This causes function to be "compiled", from tutorial
def train_step(images):

    # Random values in normal distribution between batch size and noise_dim
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    # zip function organizes gradients and variables into tuples
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start))


def generate_images(model, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.show()
