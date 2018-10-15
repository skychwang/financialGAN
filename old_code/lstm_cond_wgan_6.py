from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.merge import _Merge
from keras import regularizers
from keras.layers import *
from keras.models import load_model
from keras.optimizers import RMSprop, Adam
from read_json import *
from order_vector import *
from functools import partial
import gc
import time
from discrimination import *

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((64, 1, 1,1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class lstm_cond_gan(object):
    def __init__(self, history_ol=10, orderLength=8, historyLength=100,\
        noiseLength=1,noiseLength_1 = 100,lstm_out_length=4,mini_batch_size=1,\
        data_path=None,batch_size=64):
        self.history_ol = history_ol
        self.orderLength = orderLength
        self.historyLength = historyLength
        self.noiseLength = noiseLength
        self.noiseLength_1 = noiseLength_1
        self.lstm_out_length = lstm_out_length
        self.mini_batch_size = mini_batch_size
        self.data_path = data_path
        self.batch_size = batch_size
        self.model = None
        self.build()

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

    def attention_3d_block(self,inputs,SINGLE_ATTENTION_VECTOR=False):
        input_dim = int(inputs.shape[1])
        a = Permute((2, 1))(inputs)
        a = Dense(input_dim, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1))(a)
            a = RepeatVector(self.history_ol)(a)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = merge([inputs, a_probs], mode='mul')
        return output_attention_mul

    def build(self):
        # build models
        if self.model:
            return self.model
        # lstm cell, to do : attention mechanism
        history = Input(shape=(4,), \
            name='history_full')
        history_input = Input(shape=(1,), \
            name='history_input')
        #lstm_output = LSTM(self.lstm_out_length)(history)
        #dense_1 = Flatten()(history_input)
        #lstm_output = Dense(self.lstm_out_length)(dense_1)
        #D lstm with attention mechamism
        #lstm_output_h = LSTM(self.lstm_out_length,name='lstm_critic')(history_input)
        #lstm_output_d = Dense(self.orderLength,name='dense_critic')(lstm_output_h)


        # merge with noise
        noise_input = Input(shape=(self.noiseLength,), name='noise_input')
        noise_input_1 = Input(shape=(self.noiseLength_1,), name='noise_input_1')
        gen_input = Concatenate(axis=-1)([history_input,noise_input])
        gen_input_1 = Concatenate(axis=-1)([history,noise_input_1])
        #generator
        dropout = 0.5
        G = Sequential(name='generator')
        G.add(Dense(self.mini_batch_size * 100, input_dim=self.noiseLength+1))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        #G.add(Dense(128))
        #G.add(BatchNormalization())
        #G.add(Activation('relu'))
        #G.add(Dense(32))
        #G.add(BatchNormalization())
        #G.add(Activation('relu'))
        #G.add(Dense(8))
        #G.add(BatchNormalization())
        #G.add(Activation('relu'))
        #G.add(Dense(1))
        #G.add(BatchNormalization())
        #G.add(Activation('tanh'))
        #G.add(Reshape((int(self.mini_batch_size), 1,1)))
        G.add(Reshape((int(self.mini_batch_size), 1, 100)))
        G.add(UpSampling2D())
        G.add(Conv2DTranspose(64, 32, padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Dropout(dropout))
        G.add(UpSampling2D())
        G.add(Conv2DTranspose(32, 32, padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Conv2DTranspose(16, 32 , padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Conv2DTranspose(8, 32, padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Conv2DTranspose(4, 32, padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(MaxPooling2D((2,2)))
        G.add(Conv2DTranspose(1, 32, padding='same'))
        G.add(Activation('tanh'))
        G.add(MaxPooling2D((2,2)))
        generator_output = G(gen_input)

        dropout = 0.5
        G_1 = Sequential(name='generator_1')
        G_1.add(Dense((self.orderLength-1)*self.mini_batch_size*100, \
            input_dim=self.noiseLength_1+self.lstm_out_length))
        G_1.add(BatchNormalization())
        G_1.add(Activation('relu'))
        G_1.add(Reshape((int(self.mini_batch_size), int(self.orderLength - 1), 100)))
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
        generator_output_1 = G_1(gen_input_1)

        gen_output = Concatenate(axis=2)([generator_output,generator_output_1])

        truth_input = Input(shape=(self.mini_batch_size,self.orderLength,1),name='truth_input')
        #discriminator_input_fake = (Concatenate(axis=2)\
        #    ([Reshape((1, 1,1))(history_input), generator_output]))
        #discriminator_input_truth = Concatenate(axis=2)\
        #    ([Reshape((1, 1,1))(history_input), truth_input])
        discriminator_input_fake = gen_output
        discriminator_input_truth = truth_input

        #gradient penelty
        averaged_samples = RandomWeightedAverage()([discriminator_input_fake, discriminator_input_truth])

        #discriminator
        D = Sequential(name='discriminator')
        D.add(Conv2D(512,(3,3),padding='same',  input_shape=(self.mini_batch_size, self.orderLength,1)))
        D.add(Activation('relu'))
        D.add(Conv2D(256, (3,3),padding='same'))
        D.add(Activation('relu'))
        D.add(Conv2D(128,(3,3),padding='same'))
        D.add(Activation('relu'))
        D.add(Flatten())
        D.add(Dense(1))
        #D.add(Activation('tanh'))
        self.D = D
        discriminator_output_fake = D(discriminator_input_fake)
        discriminator_output_truth = D(discriminator_input_truth)
        averaged_samples_output = D(averaged_samples)

        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=1)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

        self.gen = Model(inputs=[history_input,history,noise_input,noise_input_1], outputs= gen_output)
        self.model_truth = Model(inputs=[history_input,history,noise_input,noise_input_1,truth_input],\
             outputs= [discriminator_output_fake,discriminator_output_truth,averaged_samples_output])
        self.model_fake = Model(inputs=[history_input,history,noise_input,noise_input_1],\
             outputs= discriminator_output_fake)
        optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)
        self.gen.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.gen.summary()
        for layer in self.model_truth.layers:
            layer.trainable = False
        self.model_truth.get_layer(name='discriminator').trainable = True
        #self.model_truth.get_layer(name='lstm_critic').trainable = True
        #self.model_truth.get_layer(name='dense_critic').trainable = True
        self.model_truth.compile(optimizer=optimizer, \
            loss=[self.w_loss,self.w_loss,partial_gp_loss])
        for layer in self.model_fake.layers:
            layer.trainable = True
        self.model_fake.get_layer(name='discriminator').trainable = False
        #self.model_fake.get_layer(name='lstm_critic').trainable = False
        #self.model_fake.get_layer(name='dense_critic').trainable = False
        self.model_fake.compile(optimizer=optimizer, loss=self.w_loss)
        self.model_fake.summary()
        self.model_truth.summary()

    def Recover_interval(self,data,ref):
        #print(np.round(ref))
        data_new = np.zeros((self.mini_batch_size,6))
        if(data.shape[1]==3):
            data_new[:,3:] =np.floor(data)
        else:
            #if(np.max(((((data[:,0]*10 + data[:,1])*10 + data[:,2])*10 + data[:,3])*10 + data[:,4])*10 + data[:,5]) < 4000):
            data_new = np.floor(data)
            #else:
            #    print("Big Step")
            #    return None


        def normalize(data):
            for i in range(5,-1,-1):
                if(data[i]>9):
                    data[i] -= 10
                    if(i > 0):
                        data[i-1] += 1
            if(((((data[0]*10 + data[1])*10 + data[2])*10 + data[3])*10 + data[4])*10 + data[5] > 210000):
                data = [0,0,0,0,0,0]
            return data

        for i in range(data_new.shape[0]):
            if(i == 0):
                data_new[i,:] = data_new[i,:] + np.round(ref)
            else:
                data_new[i,:] = data_new[i,:] + data_new[i-1,:]
            data_new[i,:] = normalize(data_new[i,:])
        return data_new


    def fit(self, train_steps=300001, buy_sell_tag=0, batch_size=64, gnr_path='gnr'):
        #self.gen = load_model('gnr_goog12_100000')
        data = np.load(self.data_path, mmap_mode='r')
        for i in range(train_steps):
            positive_y = np.ones((batch_size, 1), dtype=np.float32)
            negative_y = -positive_y
            dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
            no = np.random.normal(0, 0.05 , size=[data.shape[0],batch_size, self.noiseLength])
            no_1 = np.random.uniform(-1, 1 , size=[data.shape[0],batch_size, self.noiseLength_1])

            for j in range(100):
                ## train/fake init
                idx = np.random.randint(0, data.shape[0])
                noise = no[idx]
                noise_1 = no_1[idx]
                orderStreams_train = np.squeeze(data[idx].copy())
                d_history = orderStreams_train[:,:self.historyLength,:]
                history = (np.squeeze(np.mean((d_history[:,:,0:1] * 10 + d_history[:,:,1:2]),1)) - 11.5)/11.5
                history_full = np.mean(self.normalize(orderStreams_train[:,:self.historyLength,6:]),1)
                truth = np.expand_dims(self.normalize(orderStreams_train[:,self.historyLength:,4:]),-1)
                d_loss = self.model_truth.train_on_batch([history,history_full,noise,noise_1,\
                    truth], [negative_y,positive_y,dummy_y])


            a_loss = self.model_fake.train_on_batch([history,history_full,noise,noise_1], positive_y)
            log_mesg = "%d: [D_fake loss: %f,D_truth loss: %f] " % (i, d_loss[0],d_loss[1])
            log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss)
            with open('log_goog_new.txt','a') as f:
                f.write(log_mesg+'\n')
                f.close()
            #print(log_mesg)
            if i % 1000 == 0:
                self.gen.save(gnr_path+'_'+str(i))

    def denormalize(self, normArray):
        def denormalize_one_dim(data,maxV=1, minV=0, high=1, low=-1):
            return ((((data - high) * (maxV - minV))/(high - low)) + maxV)

        Array = normArray.copy()
        # GooG
        maxV =  [2.34,9,9,9,1,1,941.79,680]
        minV = [0.12,0,0,0,0,0,916.13,0.6]
        # Syn32
        #maxV =  [1.8,9,9,9,1,1,1,2]
        #minV = [1.5,0,0,0,0,0,-1,0]
        # Syn64
        #maxV =  [3.8,9,9,9,1,1,1,2]
        #minV = [3.5,0,0,0,0,0,-1,0]
        #PN
        #maxV =  [46.5,9,9,9,1,1,12.68,635]
        #minV = [12.7,0,0,0,0,0,6.41,0.82]
        #PN
        #maxV =  [46.5,9,9,9,1,1,12.68,635]
        #minV = [6.54,0,0,0,0,0,6.41,0.8]
        for i in range(Array.shape[2]):
            Array[:,:,i] = denormalize_one_dim(normArray[:,:,i],maxV=maxV[i],minV=minV[i])
        newArray = np.zeros((Array.shape[0],Array.shape[1],6))
        newArray[:,:,0] = Array[:,:,0]
        newArray[:,:,2:] = Array[:,:,4:]
        newArray[:,:,1] = (Array[:,:,1] * 10 + Array[:,:,2]) * 10 + Array[:,:,3]
        return newArray

    def normalize(self, normArray):
        def normalize_one_dim(array, maxV=1, minV=0, high=1, low=-1):
            return (high - (((high - low) * (maxV - array)) / (maxV - minV)))

        Array = normArray.copy()
        # GooG
        maxV =  [2.34,9,9,9,1,1,941.79,680]
        minV = [0.12,0,0,0,0,0,916.13,0.6]
        # Syn32
        #maxV =  [1.8,9,9,9,1,1,1,2]
        #minV = [1.5,0,0,0,0,0,-1,0]
        # Syn64
        #maxV =  [3.8,9,9,9,1,1,1,2]
        #minV = [3.5,0,0,0,0,0,-1,0]
        #PN
        #maxV =  [46.5,9,9,9,1,1,12.68,635]
        #minV = [12.7,0,0,0,0,0,6.41,0.82]
        #PN
        #maxV =  [46.5,9,9,9,1,1,12.68,635]
        #minV = [6.54,0,0,0,0,0,6.41,0.8]
        if Array.shape[2] == 4:
            for i in range(Array.shape[2]):
                Array[:,:,i] = normalize_one_dim(normArray[:,:,i],maxV=maxV[i+4],minV=minV[i+4])
            return Array
        else:
            newArray = np.zeros((Array.shape[0],Array.shape[1],8))
            newArray[:,:,0] = Array[:,:,0]
            newArray[:,:,4:] = Array[:,:,2:]
            re = Array[:,:,1].astype(int)
            newArray[:,:,3] = re % 10
            re = re // 10
            newArray[:,:,2] = re % 10
            newArray[:,:,1] = re // 10
            for i in range(newArray.shape[2]):
                newArray[:,:,i] = normalize_one_dim(newArray[:,:,i],maxV=maxV[i],minV=minV[i])
            return newArray



    def predict(self,save_path='predict_goog_short_5000.npy',length=4000000,step_size=1,num_runs=1):
        def translate_time(time):
            transfered_time = []
            for _ in range(6):
                r = time % 10
                transfered_time.insert(0,r)
                time = time // 10
            return transfered_time

        data = np.load(self.data_path, mmap_mode='r')
        gen = load_model('gnr_goog_short_5000')
        #re = np.zeros((num_runs,length*step_size+self.historyLength,2))
        generated_orders = np.zeros((num_runs, length*step_size+self.historyLength,6))
        for j in range(num_runs):
            #history = self.normalize(data[0,0:1,:self.historyLength,:,0])
            history = np.ones((1,1))*(0-11.5)/11.5
            history_full = np.mean(self.normalize(data[0,0:1,:100,6:,0]),1)
            #generated_orders[j,:self.historyLength,:] =  self.denormalize(history)
            for i in range(length):
                noise = np.random.normal(0,0.05,size=[1, self.noiseLength])
                noise_1 = np.random.uniform(-1,1,size=[1, self.noiseLength_1])
                result = np.zeros((1,1,1))
                for k in range(1):
                    orderStreams = self.denormalize(np.squeeze(gen.predict([history,history_full,noise,noise_1]),-1))
                    generated_orders[j,self.historyLength+ i * step_size : self.historyLength+(i+1)*step_size,2:] = orderStreams[:,:,2:]
                    interval = orderStreams[0,:,:2]
                    #re[j,self.historyLength + i * step_size : self.historyLength+(i+1)*step_size,:] = interval
                    result[k,:,0] =  np.round(interval[:,0] * interval[:,1])
                result_ave = np.mean(result,0)
                r = generated_orders[j:j+1,self.historyLength + i*step_size - 1,0] + result_ave
                generated_orders[j,self.historyLength + i * step_size : self.historyLength+(i+1)*step_size,0] =  r
                history = (np.floor(generated_orders[j:j+1,self.historyLength+ (i+1)*step_size -1,0]/10000) - 11.5)/11.5
                history_full = np.mean(self.normalize(generated_orders[j:j+1,(i+1)*step_size : self.historyLength+ (i+1)*step_size,2:]),1)
                if history > 1:
                    break
                if(i % 100 == 0 ):
                    print(str(j)+' runs ' + str(i)+' steps')
                    print(result_ave,history*11.5 + 11.5)
        np.save(save_path,generated_orders)
        #np.save('interval.npy',re)
