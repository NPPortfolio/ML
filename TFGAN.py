import tensorflow as tf
import tensorflow_gan as tfgan

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
        images = (tf.cast(element['image'], tf.float32) - 127.5) / 127.5
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

params = {'batch_size':100, 'noise_dims':64}