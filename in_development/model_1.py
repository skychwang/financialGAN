# Real data tests

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential,Model
from keras import regularizers
from keras.layers import (AveragePooling2D,LeakyReLU,Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Input, Activation, Flatten, Dense, Reshape, UpSampling2D, BatchNormalization)
from keras.optimizers import RMSprop, Adam
from randomOrderGenerator import (randomOrderGenerator, randomLabelGenerator)
from read_json import *
from order_vector import *
import scipy.ndimage as ndimage

import pdb

class GAN(object):
	def __init__(self, orderStreamSize=10, orderLength=600):
		# Orderstream dimensions init
		self.orderStreamSize = orderStreamSize
		self.orderLength = orderLength

		# init
		self.D = None
		self.G = None

		optimizer = Adam(0.00001)

		self.discriminator = self.discriminator()
		self.discriminator.compile(loss='binary_crossentropy', 
						optimizer=optimizer,
						metrics=['accuracy'])

		self.generator = self.generator()
		optimizer_1 = Adam(0.075)
		#self.generator.compile(loss=self.sparse_loss(img), optimizer=optimizer_1)

		z = Input(shape=(100,))
		img = self.generator(z)
		valid = self.discriminator(img)
		self.combined = Model(z,valid)
		self.combined.compile(loss=self.sparse_loss(img), optimizer=optimizer_1)

	def sparse_loss(self,img):
		def loss(y_true,y_pred):
			alpha =2 
			beta = 50
			gamma = 5
			img_loss = (K.mean(K.abs(((img/2+0.5)**beta)/((img/2+0.5)**beta+(img/2+0.5)**gamma+1))))
			cross_entropy_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
			return alpha*img_loss + cross_entropy_loss
		return loss

	def sparse_loss_2(self,img):
		def loss(y_true,y_pred):
			alpha =2
			beta = 50
			gamma = 5
			epsilon =1e-8 
			img_loss =(K.mean(K.abs(((img/2+0.5)**beta)/((img/2+0.5)**beta+(img/2+0.5)**gamma+1))))
			b_loss = 0.5 * K.mean((K.log(epsilon+y_pred) - K.log(epsilon+1- y_pred))**2,axis=-1)
			return alpha*img_loss + b_loss
		return loss

	def discriminator(self):
		if self.D:
			return self.D

		# discriminator vars init
		dropout = 0.4

		# model definition
		self.D = Sequential()
		self.D.add(Conv2D(64, (3, 3), activation='elu', padding='same',input_shape=(self.orderStreamSize, self.orderLength, 1)))
		#self.D.add(MaxPooling2D(pool_size=(2, 2)))
		self.D.add(Dropout(dropout))
		self.D.add(Conv2D(64, (3, 3), activation='elu',padding='same'))
		#self.D.add(MaxPooling2D(pool_size=(2, 2)))
		self.D.add(Dropout(dropout))

		self.D.add(Conv2D(32,(3,3), activation='elu',padding='same'))
		self.D.add(MaxPooling2D(pool_size=(2,2)))
		self.D.add(Dropout(dropout))


		# discriminator output - 1-dimensional probability
		self.D.add(Flatten())
		self.D.add(Dense(1))
		self.D.add(Activation('sigmoid'))

		self.D.summary()

		return self.D

	def generator(self):
		if self.G:
			return self.G

		# generator input - uniform distribution, size (100)

		# generator vars init
		dropout = 0.6

		# generator definition, v4, variable orderstream size, tested with (10, 600)
		self.G = Sequential()
		self.G.add(Dense(self.orderStreamSize*self.orderLength, input_dim=100))#self.G.add(Dense(10*5*400, input_dim=100))
		self.G.add(BatchNormalization())
		self.G.add(LeakyReLU())
		self.G.add(Reshape((int(self.orderStreamSize/5), int(self.orderLength/5),25)))
		self.G.add(Dropout(dropout))

		self.G.add(UpSampling2D(size=5))
		self.G.add(Conv2DTranspose(32, 5, padding='same'))
		self.G.add(BatchNormalization())
		self.G.add(LeakyReLU())
		self.G.add(Activation('relu'))
		self.G.add(UpSampling2D(size=3))
		self.G.add(Conv2DTranspose(8, 5, padding='same'))
		self.G.add(BatchNormalization())
		self.G.add(LeakyReLU())
		self.G.add(Conv2DTranspose(4, 5, padding='same'))
		self.G.add(BatchNormalization())
		self.G.add(LeakyReLU())

		# generator output - orderstream, size (orderStreamSize, orderLength, 1)
		self.G.add(Conv2D(2, 5, padding='same'))
		self.G.add(AveragePooling2D(pool_size=(3,3)))
		self.G.add(LeakyReLU())
		self.G.add(Conv2D(1,5, padding='same'))
		#self.G.add(AveragePooling2D(pool_size=(2,2)))
		self.G.add(Activation('tanh'))

		self.G.summary()

		return self.G
class financial_GAN(object):
	def __init__(self):
		# generated orderstream dimensions
		self.orderStreamSize = 10
		self.orderLength = 600

		# init
		self.GAN = GAN(orderStreamSize=self.orderStreamSize, orderLength=self.orderLength)
		self.discriminator = self.GAN.discriminator
		self.generator = self.GAN.generator
		self.combined = self.GAN.combined

	def normalize(self, array, maxV=np.amax(np.load("data.npy", mmap_mode='r')), minV=0, high=1, low=-1):
		a = 1.1
		return   a-(a+1)*np.exp(-(array/maxV)**0.5*np.log((a+1)/(a-1)))

	def denormalize(self, normArray, maxV=np.amax(np.load("data.npy", mmap_mode='r')), minV=0, high=1, low=-1):
		a = 1.1
		return (np.log((a-normArray)/(a+1))/np.log((a+1)/(a-1)))**2*maxV


	def train(self, train_steps=100, batch_size=256, pretrain_size=5000):
		# Pretrain 10 epochs
		'''	
		xstartingidx = np.random.randint(0, np.load("data.npy", mmap_mode='r').shape[0] - pretrain_size)
		idx = np.linspace(xstartingidx, xstartingidx+pretrain_size, pretrain_size, endpoint=False, dtype=int) 
		x = self.normalize(np.load("data.npy", mmap_mode='r')[idx])#self.normalize(next(randomOrderGenerator(pretrain_size, self.orderStreamSize, self.orderLength)))
		y = np.ones([pretrain_size, 1])
		print("\nPretrain of discriminator:\n")
		self.discriminator.fit(x, y, epochs=10)
		'''

		# USES TRAIN_ON_BATCH
		# Real and Generated train
		
		for i in range(train_steps):
			## gen noise init
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			## train/fake init
			idx = np.random.randint(0, np.load("data.npy", mmap_mode='r').shape[0])
			orderStreams_train = self.normalize(np.load("data.npy", mmap_mode='r')[idx])
			orderStreams_fake = self.generator.predict(noise) # effectively concats generated to integers.
			## data/labels init
			x = np.concatenate((orderStreams_train[:,:,:,0].reshape(batch_size,10,600,1), orderStreams_fake))
			y = np.ones([batch_size*2, 1])
			y[batch_size:, :] = 0
			## D training
			d_loss = self.discriminator.train_on_batch(x, y)
			self.discriminator.trainable = False
			y = np.ones([batch_size, 1])
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			a_loss = self.combined.train_on_batch(noise,y)
			self.discriminator.trainable = True
			# output
			log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
			log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss*100)
			print(log_mesg)
			if i % 10 == 0:
				self.discriminator.save_weights('discriminator', True)
				self.generator.save_weights('generator', True)

		generator =self.denormalize(x)
		np.save('gen.npy',generator)
		

if __name__ == '__main__':
	fingan = financial_GAN()
	fingan.train()
