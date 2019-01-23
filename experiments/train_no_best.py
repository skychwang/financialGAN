import time

from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers.merge import _Merge
from keras.layers import *
from keras.optimizers import RMSprop, Adam

from functools import partial
from discrimination import *


# this is following Improved WGAN
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((64, 1, 1,1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class lstm_cond_gan(object):
    """ This defines the GAN for training stock market simulator
    Arguments:
    orderLength: integer, length of one order (all features of order such as price, quantity, time diff(2), types, best bid/ask(4))
    historyLength: integer, number of past orders used as history
    noiseLength: integer, length of noise vector
    lstm_out_length: integer, length of embedding vector of history
    mini_batch_size: integer, number of orders generated each time
    data_path: string, path of training dataset
    batch_size: integer, number of orders within one batch
    """
    def __init__(self, orderLength=5, historyLength=20,\
            noiseLength=100, lstm_out_length=9, mini_batch_size=1,\
        data_path=None, batch_size=64):
        self.orderLength = orderLength
        self.historyLength = historyLength
        self.noiseLength = noiseLength
        self.lstm_out_length = lstm_out_length
        self.mini_batch_size = mini_batch_size
        self.data_path = data_path
        self.batch_size = batch_size
        self.model = None
        self.build()

	# this is following Improved WGAN
    def gradient_penalty_loss(self,y_true, y_pred, averaged_samples, \
        gradient_penalty_weight):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def w_loss(self,y_true,y_pred):
        return K.mean(y_true*y_pred)

    def build(self):
        # build models

        if self.model:
            return self.model

        ##################### Input for both Generator and Critic ############################################
        # history orders of shape (self.historyLength, self.orderLength)
        history = Input(shape=(self.historyLength, self.orderLength), \
            name='history_full')
        # current time slot: Integer, from 0 to 23
        history_input = Input(shape=(1,), name='history_time')
        # noise input of shape (self.noiseLength)
        noise_input_1 = Input(shape=(self.noiseLength,), name='noise_input_1')

        # Real order of shape((self.mini_batch_size,self.orderLength)
        truth_input = Input(shape=(self.mini_batch_size,self.orderLength,1),name='truth_input')


        # lstm at Generator to extract history orders features
        lstm_output = LSTM(self.lstm_out_length)(history)

        # lstm at Critic to extract history orders features
        lstm_output_h = LSTM(self.lstm_out_length,name='lstm_critic')(history)

        # concatenate history features with noise
        gen_input = Concatenate(axis=-1)([history_input,lstm_output,noise_input_1])

        ####################### Generator ######################################
        # Input: gen_input, shape(self.noiseLength+self.lstm_out_length + 1)
        # Output: gen_output_1, shape(self.mini_batch_size,self.orderLength)
        dropout = 0.5
        G_1 = Sequential(name='generator_1')
        G_1.add(Dense((self.orderLength)*self.mini_batch_size*100, \
            input_dim=self.noiseLength+self.lstm_out_length + 1))
        G_1.add(BatchNormalization())
        G_1.add(Activation('relu'))
        G_1.add(Reshape((int(self.mini_batch_size), int(self.orderLength), 100)))
        G_1.add(UpSampling2D())
        G_1.add(Dropout(dropout))
        G_1.add(UpSampling2D())
        G_1.add(Conv2DTranspose(32, 32, padding='same'))
        G_1.add(BatchNormalization())
        G_1.add(Activation('relu'))
        G_1.add(Conv2DTranspose(16,32 , padding='same'))
        G_1.add(BatchNormalization())
        G_1.add(Activation('relu'))
        G_1.add(Conv2DTranspose(8, 32, padding='same'))
        G_1.add(BatchNormalization())
        G_1.add(Activation('relu'))
        G_1.add(MaxPooling2D((2,2)))
        G_1.add(Conv2DTranspose(1, 32, padding='same'))
        G_1.add(Activation('tanh'))
        G_1.add(MaxPooling2D((2,2)))

        #Output of Generator, shape(self.mini_batch_size, self.orderLength)
        gen_output = G_1(gen_input)

        ##################### Critic ###########################################
        # Input of Critic, merge history_input, lstm_output_h and gen_output/truth_input
        discriminator_input_fake = (Concatenate(axis=2)\
            ([Reshape((1, 1,1))(history_input), \
            Reshape((1, self.lstm_out_length,1))(lstm_output_h), gen_output]))
        discriminator_input_truth = Concatenate(axis=2)\
            ([Reshape((1, 1,1))(history_input), \
            Reshape((1, self.lstm_out_length,1))(lstm_output_h), truth_input])
        #random-weighted average of real and generated samples - following Improved WGAN work
        averaged_samples = RandomWeightedAverage()\
            ([discriminator_input_fake, discriminator_input_truth])

        #Critic
        #Input: discriminator_input_fake/discriminator_input_truth
        #Ouput: score
        D = Sequential(name='discriminator')
        D.add(Conv2D(512,(3,3),padding='same',  input_shape=(self.mini_batch_size, \
                self.orderLength+self.lstm_out_length+1,1)))
        D.add(Activation('relu'))
        D.add(Conv2D(256, (3,3),padding='same'))
        D.add(Activation('relu'))
        D.add(Conv2D(128,(3,3),padding='same'))
        D.add(Activation('relu'))
        D.add(Flatten())
        D.add(Dense(1))
        #self.D = D

        discriminator_output_fake = D(discriminator_input_fake)
        discriminator_output_truth = D(discriminator_input_truth)
        averaged_samples_output = D(averaged_samples)

        #Def gradient penalty loss
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=1)
        partial_gp_loss.__name__ = 'gradient_penalty'

        ########################### Model Definition  ##########################
        # Generator model
        # Input: [history_input,history,noise_input_1]
        # Output: gen_output
        self.gen = Model(inputs=[history_input,history,noise_input_1], outputs= gen_output)
        #Model Truth:
        self.model_truth = Model(inputs=[history_input,history,noise_input_1,truth_input],\
             outputs= [discriminator_output_fake,discriminator_output_truth,averaged_samples_output])
        #Model Fake:
        self.model_fake = Model(inputs=[history_input,history,noise_input_1],\
             outputs= discriminator_output_fake)
        #Optimizer
        optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)

        #Compile Models
        #Generator
        self.gen.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.gen.summary()
        #Model Truth - Generator is not trainable here
        for layer in self.model_truth.layers:
            layer.trainable = False
        self.model_truth.get_layer(name='discriminator').trainable = True
        self.model_truth.get_layer(name='lstm_critic').trainable = True
        self.model_truth.compile(optimizer=optimizer, \
            loss=[self.w_loss,self.w_loss,partial_gp_loss])
        #Model Fake - critic is not trainable here
        for layer in self.model_fake.layers:
            layer.trainable = True
        self.model_fake.get_layer(name='discriminator').trainable = False
        self.model_fake.get_layer(name='lstm_critic').trainable = False
        self.model_fake.compile(optimizer=optimizer, loss=self.w_loss)
        #print summary
        self.model_fake.summary()
        self.model_truth.summary()

	# gnr_path = path to save generator model
    def fit(self, train_steps=300001, batch_size=64, gnr_path='gnr'):
        #import data
        data = np.load(self.data_path, mmap_mode='r')


        for i in range(train_steps):
			# postive_y and negative_y go ultimately into the loss functions
            positive_y = np.ones((batch_size, 1), dtype=np.float32)
            negative_y = -positive_y
			# dummy_y goes as y_true in gradient_penalty_loss
            dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
            noise = np.random.uniform(-1, 1 , size=[data.shape[0],batch_size, self.noiseLength])

            for j in range(100): # critic trained 100 times
                #Get one sample from index
                idx = np.random.randint(0, data.shape[0])
                # Get Noise
                noise_1 = noise[idx]

                ### Prepare Data
                #Normalization
                orderStreams_train = self.normalize(np.squeeze(data[idx].copy()))
                #History time
                history = orderStreams_train[:,self.historyLength-1,0:1]
                #History
                history_full = orderStreams_train[:,:self.historyLength,1:6]
                #Real orders
                truth = np.expand_dims(orderStreams_train[:,self.historyLength:,1:6],-1)

                #Train Critic
                d_loss = self.model_truth.train_on_batch([history,history_full,noise_1,\
                    truth], [negative_y,positive_y,dummy_y])

            #Train Generator
            a_loss = self.model_fake.train_on_batch([history,history_full,noise_1], positive_y)

            #Logging
            log_mesg = "%d: [D_fake loss: %f,D_truth loss: %f] " % (i, d_loss[0],d_loss[1])
            log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss)
            with open('log_goog_no_best.txt','a') as f:
                f.write(log_mesg+'\n')
                f.close()
            if i % 1000 == 0:
                self.gen.save(gnr_path+'_'+str(i))

    def denormalize(self, normArray):
        def denormalize_one_dim(data,maxV=1, minV=0, high=1, low=-1):
            return ((((data - high) * (maxV - minV))/(high - low)) + maxV)

        Array = normArray.copy()
        # 10 dims: [inter-arrival time;buy/sell;cancel/not cancel/;price;
        #           quantity;best bid price;best ask price;best bid quantity;
        #           best ask quantity]
        # MinMax Values for different dataset
        #New_rep
        maxV =  [16500,1,1,942,150,942,942,3000,3000]
        minV = [0,0,0,916,0,916,916,1,1]
        # GooG
        #maxV =  [205,9,9,9,1,1,942,620]
        #minV = [4,0,0,0,0,0,916,0]
        # Syn32
        #maxV =  [63,1,1,1,1,1,1,2,2]
        #minV = [0,0,0,-1,-1,-1,-1,1,1]
        # Syn64
        #maxV =  [3.8,9,9,9,1,1,1,2]
        #minV = [3.5,0,0,0,0,0,-1,0]
        #PN
        #maxV =  [np.log(233448 + 1),1,1,13,300,13,13,40000,40000]
        #minV = [0,0,0,6,0,6,6,1,1]
        #maxV =  [233448,1,1,13,300,13,13,40000,40000]
        #minV = [0,0,0,6,0,6,6,1,1]
        #PN
        #maxV =  [46.5,9,9,9,1,1,12.68,635]
        #minV = [6.54,0,0,0,0,0,6.41,0.8]

        for i in range(Array.shape[2]):
            Array[:,:,i] = denormalize_one_dim(normArray[:,:,i],maxV=maxV[i],minV=minV[i])

        return Array

    def normalize(self, normArray):
        def normalize_one_dim(array, maxV=1, minV=0, high=1, low=-1):
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if array[i,j] < minV:
                        array[i,j] = minV
                    if array[i,j] > maxV:
                        array[i,j] = maxV
            return (high - (((high - low) * (maxV - array)) / (maxV - minV)))

        Array = normArray.copy()

        #MinMax Values for different dataset
        # GooG
        maxV =  [23,16500,1,1,942,150,942,942,3000,3000]
        minV = [0,0,0,0,916,0,916,916,1,1]
        # Syn32
        #maxV =  [1,63,1,1,1,1,1,1,2,2]
        #minV = [0,0,0,0,-1,-1,-1,-1,1,1]
        # Syn64
        #maxV =  [3.8,9,9,9,1,1,1,2]
        #minV = [3.5,0,0,0,0,0,-1,0]
        #PN
        #maxV =  [23,np.log(233448 + 1),1,1,13,300,13,13,40000,40000]
        #minV = [0,0,0,0,6,0,6,6,1,1]
        #maxV =  [1,233448,1,1,13,300,13,13,40000,40000]
        #minV = [0,0,0,0,6,0,6,6,1,1]
        #PN
        #maxV =  [46.5,9,9,9,1,1,12.68,635]
        #minV = [6.54,0,0,0,0,0,6.41,0.8]

        for i in range(Array.shape[2]):
            Array[:,:,i] = normalize_one_dim(normArray[:,:,i],maxV=maxV[i],minV=minV[i])
        return Array

	# length is the maximum number of orders outputted in 1 run
    def predict(self,save_path='predict_no_best.npy',length=600000,step_size=1,num_runs=1):

        #Load Data
        data = np.load(self.data_path, mmap_mode='r')
        #Load Generator
        gen = load_model('gnr_no_best_30000')

        #Allocate space for generated orders
        generated_orders = np.zeros((num_runs, length*step_size+self.historyLength,6))


        for j in range(num_runs):
            #Get seeds from real data
            da = data[0,0:1,:,:,0].copy()
            orderStreams_train = self.normalize(da)
            history = orderStreams_train[:,self.historyLength-1,0:1]
            history_full = orderStreams_train[:,:self.historyLength,1:6]
            generated_orders[j,:self.historyLength,1:] =  self.denormalize(history_full)

            for i in range(length):
                noise_1 = np.random.uniform(-1,1,size=[1, self.noiseLength])
                orderStreams = self.denormalize(np.squeeze(gen.predict([history,history_full,noise_1]),-1))

                generated_orders[j,self.historyLength+ i * step_size : self.historyLength+(i+1)*step_size,1:]\
                            = orderStreams
                r = generated_orders[j:j+1,self.historyLength + i*step_size - 1,0] + orderStreams[0,:,:1]
                generated_orders[j,self.historyLength + i * step_size : self.historyLength+(i+1)*step_size,0] =  r

				# 11.5 corresponds to 23 time slots
                history = (np.floor(generated_orders[j:j+1,self.historyLength+ (i+1)*step_size -1,0]/1000000) - 11.5)/11.5
                history_full = self.normalize(generated_orders[j:j+1,(i+1)*step_size : self.historyLength+ (i+1)*step_size,:].copy())[:,:,1:]

                #stop after generating one day long stream
                if history > 1:
                    break

                #print some stats
                if(i % 1000 == 0 ):
                    print(str(j)+' runs ' + str(i)+' steps')
                #    print(interval,history*11.5 + 11.5)
        #save generated orders
        np.save(save_path,generated_orders)
