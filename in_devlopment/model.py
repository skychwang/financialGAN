import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras import regularizers
from keras.layers import (Conv2D, MaxPooling2D, Dropout, Input, Activation, Flatten, Dense, Reshape, UpSampling2D, BatchNormalization)
from keras.optimizers import RMSprop
from randomOrderGenerator import (randomOrderGenerator, randomLabelGenerator)

class GAN(object):
	def __init__(self, orderLength=50, orderStreamSize=100, filterSize=(3, 3), poolSize=(2,2), numFilters=32, dropout=0.1):
		self.orderLength = orderLength
		self.orderStreamSize = orderStreamSize
		self.filterSize = filterSize
		self.poolSize = poolSize
		self.numFilters = numFilters
		self.dropout = dropout

		self.D = None
		self.G = None
		self.AM = None
		self.DM = None

	def discriminator(self):
		if self.D:
			return self.D

		# model definition
		self.D = Sequential()
		self.D.add(Conv2D(self.numFilters, self.filterSize, activation='relu', input_shape=(self.orderLength, self.orderStreamSize, 1)))
		self.D.add(MaxPooling2D(pool_size=self.poolSize))
		self.D.add(Dropout(self.dropout))

		# GAN discriminator
		self.D.add(Flatten())
		self.D.add(Dense(1))
		self.D.add(Activation('sigmoid'))

		self.D.summary()

		return self.D

	def discriminator_model(self):
		if self.DM:
			return self.DM

		optimizer = RMSprop(lr=0.0002, decay=6e-8)
		self.DM = Sequential()
		self.DM.add(self.discriminator())
		self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.DM

	def generator(self):
		if self.G:
			return self.G

		# In: uniform dist size (100)
		# Out: order stream size (orderStreamSize, orderLength, 1)

		# generator definition
		self.G = Sequential()
		self.G.add(Dense(1, input_dim=100))
		self.G.add(BatchNormalization())
		self.G.add(Activation('relu'))
		self.G.add(Reshape((1, 1, 1)))
		self.G.add(Dropout(self.dropout))

		self.G.add(UpSampling2D(size=(self.orderLength, self.orderStreamSize)))
		self.G.add(Activation('sigmoid'))

		self.G.summary()

		return self.G

	def adversarial_model(self):
		if self.AM:
			return self.AM

		optimizer=RMSprop(lr=0.0001, decay=3e-8)
		self.AM = Sequential()
		self.AM.add(self.generator())
		self.AM.add(self.discriminator())
		self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.AM


class financial_GAN(object):
	def __init__(self):
		self.orderLength = 50
		self.orderStreamSize = 100

		# TEMP for random orders
		self.numGenerate = 10000
		self.numTest = 1000
		self.x_train = next(randomOrderGenerator(self.numGenerate, self.orderStreamSize, self.orderLength))
		#self.y_train = next(randomLabelGenerator(self.numGenerate))
		#self.x_test = next(randomOrderGenerator(self.numTest, self.orderStreamSize, self.orderLength))
		#self.y_test = next(randomLabelGenerator(self.numTest))

		self.GAN = GAN()
		self.discriminator = self.GAN.discriminator_model()
		self.adversarial = self.GAN.adversarial_model()
		self.generator = self.GAN.generator()

	def train(self, train_steps=1000, batch_size=256):
		for i in range(train_steps):
			orderStreams_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			orderStreams_fake = self.generator.predict(noise)

			x = np.concatenate((orderStreams_train, orderStreams_fake))
			y = np.ones([2*batch_size, 1])
			y[batch_size:, :] = 0
			d_loss = self.discriminator.train_on_batch(x, y)

			y = np.ones([batch_size, 1])
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			a_loss = self.adversarial.train_on_batch(noise, y)

			log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
			log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
			print(log_mesg)


		#self.discriminator.fit(self.x_train, self.y_train, epochs=100, batch_size=64, validation_data=(self.x_test, self.y_test))
		#self.discriminator.save_weights('test.h5')



if __name__ == '__main__':
	fingan = financial_GAN()
	fingan.train()