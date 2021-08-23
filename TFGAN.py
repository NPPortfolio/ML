import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow.keras import layers

# TF-GAN tutorial google collab, with some changes (small for now)

def input_fn(mode, params):

    assert 'batch_size' in params
    assert 'noise_dims' in params

    bs = params['batch_size']
    nd = params['noise_dims']

    folder = './data/train' if mode == tf.estimator.ModeKeys.TRAIN else './data/test'

    shuffle = (mode == tf.estimator.ModeKeys.TRAIN)

    just_noise = (mode == tf.estimator.ModeKeys.PREDICT)

    noise_ds = (tf.data.Dataset.from_tensors(0).repeat()
                .map(lambda _: tf.random.normal([bs, nd])))

    # Explain this
    if just_noise:
        return noise_ds

    def _preprocess(element):
        # Map [0, 255] to [-1, 1].
        images = (element - 127.5) / 127.5
        return images

    images_ds = tf.keras.preprocessing.image_dataset_from_directory(
        folder,
        seed=123,
        image_size=(256, 256),
        batch_size=bs,
        label_mode=None  # Eventually changed
    ).map(_preprocess).cache().repeat()
    
    if shuffle:
        images_ds = images_ds.shuffle(
            buffer_size=10000, reshuffle_each_iteration=True)

    images_ds = (images_ds.batch(bs, drop_remainder=True)
                 .prefetch(tf.data.experimental.AUTOTUNE))

    return tf.data.Dataset.zip((noise_ds, images_ds))

params = {'batch_size':100, 'noise_dims':100}

#TODO define what generated_inputs is, right now is it just the noise?
# may not even need to use the inputs, but have to take in constructor function args ?
def discriminator_fn(generated_image, generated_inputs):

    # Test dropout a bit

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2),
              padding='same', input_shape=(256, 256, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

# noise parameter is from the TFGAN gan_model() constructor, default generator_inputs is just noise tensor?
def generator_fn(noise_tensor):

    # Have to figure out what exactly these parameters do and the best structure for the network, sort of messing around right now

    # Conv2DTranspose upsamples and performs a convolution, sort of like a "learned" upsample
    # A stride over 1 adds gaps to the input image, which does the upsampling
    # From a tutorial: "It is also good practice to use a kernel size that is a factor of the stride (e.g. double) to avoid a checkerboard pattern
    # that can sometimes be observed when upsampling.""

    # alpha at 0.2 best practice? more info

    # NOTE: I think this input shape is where each label has to have some influence
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(tensor=noise_tensor))
    model.add(layers.Dense(16 * 32 * 32, activation='relu'))
    # Transformation that maintains the mean output close to 0 and SD close to 1
    model.add(layers.BatchNormalization())
    # Question whether to have this before or after activation function
    model.add(layers.ReLU())

    model.add(layers.Reshape((32, 32, 16)))
    print(model.output.shape)

    model.add(layers.Conv2DTranspose(8, (8, 8), strides=(2, 2),
              padding='same', use_bias=False))  # 32 64 x 64 filters
    print(model.output.shape)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # model.add(layers.Conv2DTranspose(16, (12, 12), strides=(2, 2),
      #        padding='same', use_bias=False))  # 16 128 * 128 filters
    #print(model.output.shape)
    #model.add(layers.BatchNormalization())
    #model.add(layers.ReLU())

    # Important note: probably want more than one filter here, next layer defines final image
    model.add(layers.Conv2DTranspose(4, (16, 16), strides=(4, 4),
              padding='same', use_bias=False))  # 8 256 * 256 filters
    print(model.output.shape)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # This output layer has 3 256 * 256 filters (rgb), tanh makes values between -1 and 1
    model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    print(model.output.shape)

    return model


# Learning Rates
generator_lr = 0.001
discriminator_lr = 0.0002

# Assembles and manages the pieces of the whole GAN model
gan_estimator = tfgan.estimator.GANEstimator(
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    params=params,
    # Tutoial passes a function here to change the learning rate after 100 steps, this is static
    generator_optimizer=tf.keras.optimizers.Adam(generator_lr),
    discriminator_optimizer=tf.keras.optimizers.Adam(discriminator_lr, 0.5),
)



gan_estimator.train(input_fn, max_steps=100)