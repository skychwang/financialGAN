import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras import regularizers
from keras.layers import (Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Input, Activation, Flatten, Dense, Reshape, UpSampling2D, BatchNormalization)
from keras.optimizers import RMSprop, Adam
from randomOrderGenerator import (randomOrderGenerator, randomLabelGenerator)

import pdb

class WGAN(object):
	def __init__(self, orderStreamSize=100, orderLength=50):
		# Orderstream dimensions init
		self.orderStreamSize = orderStreamSize
		self.orderLength = orderLength

		# init
		self.D = None
		self.G = None
		self.DM = None
		self.GM = None
		self.AM = None

	def wasserstein_loss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)

	def discriminator(self):
		if self.D:
			return self.D

		# discriminator vars init
		dropout = 0.4

		# model definition
		self.D = Sequential()
		self.D.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.orderStreamSize, self.orderLength, 1)))
		self.D.add(MaxPooling2D(pool_size=(2, 2)))
		self.D.add(Dropout(dropout))

		# discriminator output - 1-dimensional probability
		self.D.add(Flatten())
		self.D.add(Dense(1))
		self.D.add(Activation('linear'))

		self.D.summary()

		return self.D

	def generator(self):
		if self.G:
			return self.G

		# generator vars init
		dropout = 0.4

		# generator definition, v3, optimized for OS size (100, 50)
		self.G = Sequential()
		self.G.add(Dense(10*5*400, input_dim=100))
		self.G.add(BatchNormalization())
		self.G.add(Activation('relu'))
		self.G.add(Reshape((10, 5, 400)))
		self.G.add(Dropout(dropout))

		self.G.add(UpSampling2D(size=5))
		self.G.add(Conv2DTranspose(16, 5, padding='same'))
		self.G.add(BatchNormalization())
		self.G.add(Activation('relu'))
		self.G.add(UpSampling2D())
		self.G.add(Conv2DTranspose(8, 5, padding='same'))
		self.G.add(BatchNormalization())
		self.G.add(Activation('relu'))
		self.G.add(Conv2DTranspose(4, 5, padding='same'))
		self.G.add(BatchNormalization())
		self.G.add(Activation('relu'))

		# generator output - orderstream, size (orderStreamSize, orderLength, 1)
		self.G.add(Conv2DTranspose(1, 5, padding='same'))
		self.G.add(Activation('tanh'))####

		self.G.summary()

		return self.G

	def discriminator_model(self):
		if self.DM:
			return self.DM

		optimizer = RMSprop(lr=0.00005)
		self.DM = Sequential()
		self.DM.add(self.discriminator())
		self.DM.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
		return self.DM

	def generator_model(self):
		if self.GM:
			return self.GM

		optimizer = RMSprop(lr=0.00005)
		self.GM = Sequential()
		self.GM.add(self.generator())
		self.GM.compile(loss=self.wasserstein_loss, optimizer=optimizer)
		return self.GM

	def adversarial_model(self):
		if self.AM:
			return self.AM

		# Stacked, generator model onto the discriminator
		optimizer = RMSprop(lr=0.00005)
		self.AM = Sequential()
		self.AM.add(self.generator())
		self.AM.add(self.discriminator())
		self.AM.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
		return self.AM

class financial_GAN(object):
	def __init__(self):
		# generated orderstream dimensions
		self.orderStreamSize = 100
		self.orderLength = 50
		self.D_iters = 5
		self.clip = 0.01

		# init
		self.WGAN = WGAN(orderStreamSize=self.orderStreamSize, orderLength=self.orderLength)
		self.discriminator = self.WGAN.discriminator_model()
		self.generator = self.WGAN.generator()
		self.adversarial = self.WGAN.adversarial_model()

	def normalize(self, array, maxV=100, minV=0, high=1, low=-1):
		return (high - (((high - low) * (maxV - array)) / (maxV - minV)))

	def denormalize(self, normArray, maxV=100, minV=0, high=1, low=-1):
		return ((((normArray - high) * (maxV - minV))/(high - low)) + maxV)

	def train(self, epochs=10000, batch_size=64, save_interval=50):
		data = self.normalize(next(randomOrderGenerator(10000, self.orderStreamSize, self.orderLength)))

		for epoch in range(epochs):
			for D_iter in range(self.D_iters):
				## gen noise init
				noise = np.random.normal(0, 1, size=[batch_size, 100])
				## train/fake init
				### consecutive random indexes
				x = np.random.randint(0, data.shape[0] - batch_size)
				idx = np.linspace(x, x+batch_size, batch_size, endpoint=False, dtype=int) 
				orderStreams_train = data[idx]
				orderStreams_fake = self.normalize(self.denormalize(self.generator.predict(noise)).astype(int)) # effectively concats generated to integers.
				## data/labels init
				x = np.concatenate((orderStreams_train, orderStreams_fake))
				#############
				y = -np.ones([batch_size*2, 1])
				y[batch_size:, :] = 1
				## D training
				d_loss = self.discriminator.train_on_batch(x, y)
				## D clipping
				for layer in self.discriminator.layers:
					weights = layer.get_weights()
					weights = [np.clip(w, -self.clip, self.clip) for w in weights]
					layer.set_weights(weights)

			self.discriminator.trainable = False
			noise = np.random.normal(0, 1, (batch_size, 100))
			y = np.ones([batch_size, 1])
			a_loss = self.adversarial.train_on_batch(noise, y)
			self.discriminator.trainable = True

			log_mesg = "%d: [D loss: %f, acc: %f]" % (epoch, 1 - d_loss[0], d_loss[1])
			log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss[0])
			print(log_mesg)

			print(self.denormalize(x[65]))

			if epoch % save_interval == 0:
				self.discriminator.save_weights('discriminator', True)
				self.generator.save_weights('generator', True)

if __name__ == '__main__':
	f = financial_GAN()
	f.train()