import tensorflow as tf

# tutorial source for later
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

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