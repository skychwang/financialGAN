from keras import backend as K
from keras.models import Sequential, Model
from keras import regularizers
from keras.layers import *
from keras.models import load_model
from keras.optimizers import RMSprop, Adam
from randomOrderGenerator import (randomOrderGenerator, randomLabelGenerator)
from read_json import *
from order_vector import *
import gc
import time
from discrimination import *

class lstm_cond_gan(object):
    def __init__(self, retrain=True, orderLength=600,historyLength=100,noiseLength=100,hist_noise_ratio=1,mini_batch_size=10):
        self.orderLength = orderLength
        self.historyLength = historyLength
        self.noiseLength = noiseLength
        self.hist_noise_ratio = hist_noise_ratio
        self.lstm_out_length = self.noiseLength * self.hist_noise_ratio
        self.mini_batch_size = mini_batch_size
        self.model = None
        if(retrain):
          self.D = load_model('D')
          self.gen = load_model('gen')
          self.model_truth = load_model('model_truth')
          self.model_fake = load_model('model_fake')
        else:
           self.build()
    def w_loss(self,y_true,y_pred):
        return K.mean(y_true*y_pred)
    
    def attention_3d_block(self,inputs,SINGLE_ATTENTION_VECTOR=True):
        a = Permute((2, 1))(inputs)
        a = Dense(self.historyLength, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction_1')(a)
            a = RepeatVector(self.orderLength)(a)
        a_probs = Permute((2, 1), name='attention_vec_1')(a)
        output_attention_mul = merge([inputs, a_probs], name='attention_mul_2', mode='mul')
        return output_attention_mul

    def attention_3d_block_2(self,inputs,SINGLE_ATTENTION_VECTOR=True):
        a = Permute((2, 1))(inputs)
        a = Dense(self.mini_batch_size, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reductioni_2')(a)
            a = RepeatVector(self.orderLength)(a)
        a_probs = Permute((2, 1), name='attention_vec_2')(a)
        output_attention_mul = merge([inputs, a_probs], name='attention_muli_2', mode='mul')
        return output_attention_mul

    def build(self):
        # build models
        if self.model:
            return self.model
        # lstm cell, to do : attention mechanism
        history_input = Input(shape=(self.historyLength, self.orderLength), name='history_input')
        attention_mul = self.attention_3d_block(history_input)
        lstm_output = LSTM(self.lstm_out_length)(attention_mul)

        # merge with noise
        noise_input = Input(shape=(self.noiseLength,), name='noise_input')
        gen_input = Concatenate(axis=-1)([lstm_output, noise_input])

        #generator
        dropout = 0.5
        G = Sequential(name='generator')
        G.add(Dense(self.orderLength*self.mini_batch_size, input_dim=self.noiseLength+self.lstm_out_length))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Reshape((int(self.mini_batch_size/10), int(self.orderLength/10), 100)))
        G.add(UpSampling2D(size=5))
        G.add(Conv2DTranspose(16, 5, padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Dropout(dropout))
        G.add(UpSampling2D())
        G.add(Conv2DTranspose(8, 5, padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Conv2DTranspose(4, 5, padding='same'))
        G.add(Activation('relu'))
        G.add(Conv2DTranspose(1, 5, padding='same'))
        G.add(Activation('tanh'))
        self.G = G
        generator_output = G(gen_input)

        #merge gen_output with history
        #generator_output = self.attention_3d_block_2(Reshape((self.mini_batch_size,self.orderLength))(generator_output))
        #generator_output = Reshape((self.mini_batch_size,self.orderLength,1))(generator_output)
        discriminator_input_fake = (Concatenate(axis=1)([Reshape((self.historyLength, self.orderLength,1))(history_input), generator_output]))
        truth_input = Input(shape=(self.mini_batch_size,self.orderLength,1),name='truth_input')
        discriminator_input_truth = Concatenate(axis=1)([Reshape((self.historyLength, self.orderLength,1))(history_input), truth_input])

        #discriminator
        D = Sequential(name='discriminator')
        D.add(Conv2D(16, (3,3),  input_shape=(self.mini_batch_size+self.historyLength, self.orderLength,1)))
        #D.add(BatchNormalization())
        D.add(Activation('relu'))
        #D.add(Conv2D(8, (3,3)))
        #D.add(BatchNormalization())
        #D.add(Activation('relu'))
        D.add(Conv2D(2,(3,3)))
        D.add(BatchNormalization())
        D.add(Activation('relu'))
        D.add(Flatten())
        D.add(MinibatchDiscrimination(200,5))
        D.add(Dense(1))
        #D.add(Activation('sigmoid'))
        self.D = D
        discriminator_output_fake = D(discriminator_input_fake)
        discriminator_output_truth = D(discriminator_input_truth)

        self.gen = Model(inputs=[history_input, noise_input], outputs= generator_output)
        self.model_truth = Model(inputs=[history_input, truth_input], outputs= discriminator_output_truth)
        self.model_fake = Model(inputs=[history_input, noise_input], outputs= discriminator_output_fake)
        optimizer_1 = Adam(0.00001)
        optimizer_2 = Adam(0.075)
        self.gen.compile(optimizer=optimizer_1, loss='binary_crossentropy')
        self.gen.summary()
        self.model_truth.compile(optimizer=optimizer_1, loss=self.w_loss)
        self.model_fake.get_layer(name='discriminator').trainable = False
        self.model_fake.compile(optimizer=optimizer_2, loss=self.w_loss)
        self.model_fake.summary()
        self.model_truth.summary()

    def normalize(self, array, maxV=800, minV=0, high=1, low=-1):
        #a =1.0001
        return (high - (((high - low) * (maxV - array)) / (maxV - minV)))
        #return   a-(a+1)*np.exp(-(array/maxV)**0.5*np.log((a+1)/(a-1)))

    def denormalize(self, normArray, maxV=800, minV=0, high=1, low=-1):
        #a= 1.0001
        #return (np.log((a-normArray)/(a+1))/np.log((a+1)/(a-1)))**2*maxV
        return ((((normArray - high) * (maxV - minV))/(high - low)) + maxV)


    def fit(self, train_steps=1201,batch_size=128):
        data = np.load("data_2.npy", mmap_mode='r')
        for i in range(train_steps):
            ## gen noise init
            noise = np.random.normal(0,0.1 , size=[batch_size, self.noiseLength])
            ## train/fake init
            idx = np.random.randint(0, data.shape[0])
            orderStreams_train = self.normalize(data[idx])
            orderStreams_train_history = orderStreams_train[:,:self.historyLength,:,0]
            orderStreams_train_truth = orderStreams_train[:,self.historyLength:,:,0:1]
            #time_start = time.time()
            orderStreams_fake = self.gen.predict([orderStreams_train_history, noise]) # effectively concats generated to integers.
            #self.gen.summary()
            ## D training
            #print(time.time()-time_start)
            #print(self.model_truth.get_layer(name='discriminator')._collected_trainable_wieghts)
            x = np.concatenate((orderStreams_train_truth, orderStreams_fake))
            y = np.concatenate((np.ones((batch_size,1)),-np.ones((batch_size,1))))
            history = np.concatenate((orderStreams_train_history,orderStreams_train_history))
            #time_start = time.time()
            d_loss = self.model_truth.train_on_batch([history,x], y)
            #print(time.time()-time_start)
            ## G training
            #self.D.trainable = False
            #self.model_truth.summary()
            #optimizer_1 = Adam(0.00001)
            #self.model_truth.compile(optimizer=optimizer_1, loss='binary_crossentropy',metrics=['accuracy'])
            #optimizer_2 = Adam(0.075)
            #self.model_fake.compile(optimizer=optimizer_2, loss='binary_crossentropy',metrics=['accuracy'])
            y = np.ones((batch_size,1))
            a_loss = self.model_fake.train_on_batch([orderStreams_train_history,noise], y)
            #self.D.trainable = True
            #self.model_truth.compile(optimizer=optimizer_1, loss='binary_crossentropy',metrics=['accuracy'])
            #self.model_fake.compile(optimizer=optimizer_2, loss='binary_crossentropy',metrics=['accuracy'])
            #print(time.time()-time_start)
	    # output
            log_mesg = "%d: [D loss: %f] " % (i, d_loss)
            log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss)
            print(log_mesg)
            print(np.sum(self.denormalize(x[128:])>0),np.sum(self.denormalize(x[128:])<0))
            gc.collect()
            if i % 20 == 0:
               #self.D.save('D')
               self.gen.save('gen')
               #self.model_truth.save('model_truth')
               #self.model_fake.save('model_fake')
               generator =self.denormalize(x)
               np.save('gen_'+str(i)+'.npy',generator)

    def predict(self,length=2000):
        self.gen = load_model('gen')
        data = np.load("data_2.npy", mmap_mode='r')
        idx = np.random.randint(0, data.shape[0])
        initial_predict = self.normalize(data[idx,1,:self.historyLength,:,0])
        generated_orders = initial_predict
        for i in range(length):
            noise = np.random.uniform(-1,1,size=[1, self.noiseLength])
            orderStreams = (self.gen.predict([initial_predict.reshape((1,self.historyLength,self.orderLength)), noise]))
            generated_orders = np.concatenate((generated_orders,self.denormalize(orderStreams).reshape(self.mini_batch_size,self.orderLength)))
            initial_predict = np.concatenate((initial_predict[self.mini_batch_size:,],orderStreams.reshape(self.mini_batch_size,self.orderLength)))
        np.save('predict.npy',generated_orders)


if __name__ == '__main__':
        gan = lstm_cond_gan(retrain=False)
        gan.fit()
        gan.predict()
