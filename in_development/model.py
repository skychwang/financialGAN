import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras import regularizers
from keras.layers import (Conv2D, MaxPooling2D, Dropout, Input, Activation, Flatten, Dense, Reshape, UpSampling2D, BatchNormalization)
from keras.optimizers import RMSprop
from randomOrderGenerator import (randomOrderGenerator, randomLabelGenerator)

import pdb

class GAN(object):
	def __init__(self, orderStreamSize=100, orderLength=50):
		# Orderstream dimensions init
		self.orderStreamSize = orderStreamSize
		self.orderLength = orderLength

		# init
		self.D = None
		self.G = None
		self.AM = None
		self.DM = None

	def discriminator(self):
		if self.D:
			return self.D

		# discriminator vars init
		dropout = 0.1

		# model definition
		self.D = Sequential()
		self.D.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.orderStreamSize, self.orderLength, 1)))
		self.D.add(MaxPooling2D(pool_size=(2, 2)))
		self.D.add(Dropout(dropout))

		# discriminator output - 1-dimensional probability
		self.D.add(Flatten())
		self.D.add(Dense(1))
		self.D.add(Activation('sigmoid'))

		self.D.summary()

		return self.D

	def discriminator_model(self):
		if self.DM:
			return self.DM

		optimizer = RMSprop()
		self.DM = Sequential()
		self.DM.add(self.discriminator())
		self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.DM

	def generator(self):
		if self.G:
			return self.G

		# generator input - uniform distribution, size (100)

		# generator vars init
		dropout = 0.1

		# generator definition, tempv2
		self.G = Sequential()
		self.G.add(Dense(self.orderStreamSize, input_dim=100))
		self.G.add(BatchNormalization())
		self.G.add(Reshape((self.orderStreamSize, 1, 1)))
		self.G.add(Activation('relu'))
		self.G.add(Dropout(dropout))

		# generator output - orderstream, size (orderStreamSize, orderLength, 1)
		self.G.add(UpSampling2D(size=(1, self.orderLength)))
		self.G.add(Activation('linear'))

		self.G.summary()

		return self.G

	def adversarial_model(self):
		if self.AM:
			return self.AM

		# Stacked, generator model onto the discriminator
		optimizer=RMSprop(lr=0.5)
		self.AM = Sequential()
		self.AM.add(self.generator())
		self.AM.add(self.discriminator())
		self.AM.compile(loss='binary_crossentropy', optimizer=optimizer)
		return self.AM


class financial_GAN(object):
	def __init__(self):
		self.orderStreamSize = 100
		self.orderLength = 50

		# Random orderstream
		self.numGenerate = 10000
		self.x_train = next(randomOrderGenerator(self.numGenerate, self.orderStreamSize, self.orderLength))

		# init
		self.GAN = GAN(orderStreamSize=self.orderStreamSize, orderLength=self.orderLength)
		self.discriminator = self.GAN.discriminator_model()
		self.generator = self.GAN.generator()
		self.adversarial = self.GAN.adversarial_model()

	def train(self, train_steps=1000, batch_size=256):
		for i in range(train_steps):
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			orderStreams_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
			orderStreams_fake = self.generator.predict(noise)
			x = np.concatenate((orderStreams_train, orderStreams_fake))
			y = np.ones([2*batch_size, 1])
			y[batch_size:, :] = 0
			d_loss = self.discriminator.train_on_batch(x, y)
			y = np.ones([batch_size, 1])
			self.discriminator.trainable = False
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			a_loss = self.adversarial.train_on_batch(noise, y)
			self.discriminator.trainable = True

			log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
			log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss)
			print(log_mesg)
			#print(x[257])

			if i % 10 == 0:
				self.discriminator.save_weights('discriminator', True)
				self.generator.save_weights('generator', True)

if __name__ == '__main__':
	fingan = financial_GAN()
	fingan.train()
