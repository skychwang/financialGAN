import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras import regularizers
from keras.layers import (Conv2D, MaxPooling2D, Dropout, Input, Activation, Flatten, Dense)
from keras.optimizers import RMSprop
from randomOrderGenerator import (randomOrderGenerator, randomLabelGenerator)

class GAN(object):
	def __init__(self, orderlength=50, orderStreamSize=100, filterSize=(3, 3), poolSize=(2,2), numFilters=32, dropout=0.1):
		self.orderlength = orderlength
		self.orderStreamSize = orderStreamSize
		self.filterSize = filterSize
		self.poolSize = poolSize
		self.numFilters = numFilters
		self.dropout = dropout

		self.D = None
		#self.G = None
		#self.AM = None
		self.DM = None

	def discriminator(self):
		if self.D:
			return self.D

		# model definition
		self.D = Sequential()
		self.D.add(Conv2D(self.numFilters, self.filterSize, activation='relu', input_shape=(self.orderlength, self.orderStreamSize, 1)))
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

		optimizer = RMSprop()
		self.DM = Sequential()
		self.DM.add(self.discriminator())
		self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.DM

class financial_GAN(object):
	def __init__(self):
		self.orderlength = 50
		self.orderStreamSize = 100

		# TEMP for random orders
		self.numGenerate = 1000
		self.numTest = 100
		self.x_train = next(randomOrderGenerator(self.numGenerate, self.orderStreamSize, self.orderlength))
		self.y_train = next(randomLabelGenerator(self.numGenerate))
		self.x_test = next(randomOrderGenerator(self.numTest, self.orderStreamSize, self.orderlength))
		self.y_test = next(randomLabelGenerator(self.numTest))

		self.GAN = GAN()
		self.discriminator = self.GAN.discriminator_model()
		#self.adversarial = self.GAN.adversarial_model()
		#self.generator = self.GAN.generator()

	def train(self, train_steps=1000, batch_size=256, save_interval=0):
		
		#Future stuff for generator
		#noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
		
		self.discriminator.fit(self.x_train, self.y_train, epochs=1000, batch_size=64, validation_data=(self.x_test, self.y_test))
		self.discriminator.save_weights('test.h5')



if __name__ == '__main__':
	fingan = financial_GAN()
	fingan.discriminator.train()
