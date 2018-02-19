import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras import regularizers
from keras.layers import (Conv2D, MaxPooling2D, Dropout, Input, Activation, Flatten, Dense, Reshape, UpSampling2D, BatchNormalization)
from keras.optimizers import RMSprop, Adam
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

		optimizer = Adam()
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
		optimizer=Adam()
		self.AM = Sequential()
		self.AM.add(self.generator())
		self.AM.add(self.discriminator())
		self.AM.compile(loss='binary_crossentropy', optimizer=optimizer)
		return self.AM


class financial_GAN(object):
	def __init__(self):
		# generated orderstream dimensions
		self.orderStreamSize = 100
		self.orderLength = 50

		# init
		self.GAN = GAN(orderStreamSize=self.orderStreamSize, orderLength=self.orderLength)
		self.discriminator = self.GAN.discriminator_model()
		self.generator = self.GAN.generator()
		self.adversarial = self.GAN.adversarial_model()

	def normalize(self, array, maxV=100, minV=0, high=1, low=-1):
		return (high - (((high - low) * (maxV - array)) / (maxV - minV)))

	def denormalize(self, normArray, maxV=100, minV=0, high=1, low=-1):
		return ((((normArray - high) * (maxV - minV))/(high - low)) + maxV)


	def train(self, train_steps=1000, batch_size=1024, pretrain_size=10000):
		# Pretrain 10 epochs
		x = self.normalize(next(randomOrderGenerator(pretrain_size, self.orderStreamSize, self.orderLength)))
		y = np.ones([pretrain_size, 1])
		print("\nPretrain of discriminator:\n")
		self.discriminator.fit(x, y, epochs=10)

		# USES TRAIN_ON_BATCH
		# Real and Generated train
		"""
		for i in range(train_steps):
			## gen noise init
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			## train/fake init
			orderStreams_train = self.normalize(next(randomOrderGenerator(batch_size, self.orderStreamSize, self.orderLength)))
			orderStreams_fake = self.generator.predict(noise)
			## data/labels init
			x = np.concatenate((orderStreams_train, orderStreams_fake))
			y = np.ones([batch_size*2, 1])
			y[batch_size:, :] = 0
			## D training
			d_loss = self.discriminator.train_on_batch(x, y)
			## A=G+D training
			self.discriminator.trainable = False
			y = np.ones([batch_size, 1])
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			a_loss = self.adversarial.train_on_batch(noise, y)
			self.discriminator.trainable = True
			# output
			log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
			log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss)
			print(log_mesg)

			if i % 10 == 0:
				self.discriminator.save_weights('discriminator', True)
				self.generator.save_weights('generator', True)
		"""

		# USES FIT
		# Real and Generated train
		for i in range(train_steps):
			print("\nRound: " + str(i))
			## gen noise init
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			## train/fake init
			orderStreams_train = self.normalize(next(randomOrderGenerator(batch_size, self.orderStreamSize, self.orderLength)))
			orderStreams_fake = self.generator.predict(noise)
			## data/labels init
			x = np.concatenate((orderStreams_train, orderStreams_fake))
			y = np.ones([batch_size*2, 1])
			y[batch_size:, :] = 0
			## D training
			print("Discriminator train: ")
			d_loss = self.discriminator.fit(x, y, epochs=1, shuffle=True)
			## A=G+D training
			self.discriminator.trainable = False
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			y = np.ones([batch_size, 1])
			print("Adversarial train: ")
			a_loss = self.adversarial.fit(noise, y, epochs=1)
			self.discriminator.trainable = True

			if i % 10 == 0:
				self.discriminator.save_weights('discriminator', True)
				self.generator.save_weights('generator', True)

if __name__ == '__main__':
	fingan = financial_GAN()
	fingan.train()
