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
        weights = K.random_uniform((32, 1, 1,1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

#Q-GAN
class lstm_cond_gan(object):
    def __init__(self, history_ol=5, orderLength=5, historyLength=100,\
        noiseLength=100,hist_noise_ratio=1,mini_batch_size=1,\
        data_path=None,batch_size=32):
        self.history_ol = history_ol
        self.orderLength = orderLength
        self.historyLength = historyLength
        self.noiseLength = noiseLength
        self.hist_noise_ratio = hist_noise_ratio
        self.lstm_out_length = int(self.noiseLength * self.hist_noise_ratio)
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
        history_input = Input(shape=(self.historyLength, self.history_ol), \
            name='history_input')
        lstm_output = LSTM(self.lstm_out_length)(history_input)
        #D lstm with attention mechamism
        lstm_output_h = LSTM(self.lstm_out_length,name='lstm_critic')(history_input)
        #lstm_output_d = Dense(self.orderLength,name='dense_critic')(lstm_output_h)


        # merge with noise
        noise_input = Input(shape=(self.noiseLength,), name='noise_input')
        gen_input = Concatenate(axis=-1)([lstm_output,noise_input])

        #generator
        dropout = 0.5
        G = Sequential(name='generator')
        G.add(Dense(self.orderLength*self.mini_batch_size*100, \
            input_dim=self.noiseLength+self.lstm_out_length))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Reshape((int(self.mini_batch_size), int(self.orderLength), 100)))
        G.add(UpSampling2D())
        G.add(Conv2DTranspose(16, 32, padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Dropout(dropout))
        G.add(UpSampling2D())
        G.add(Conv2DTranspose(8, 32, padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Conv2DTranspose(4, 32, padding='same'))
        G.add(Activation('relu'))
        G.add(MaxPooling2D((2,2)))
        G.add(Conv2DTranspose(1, 32, padding='same'))
        G.add(Activation('tanh'))
        G.add(MaxPooling2D((2,2)))
        self.G = G
        generator_output = G(gen_input)

        discriminator_input_fake = (Concatenate(axis=2)\
            ([Reshape((1, self.lstm_out_length,1))(lstm_output_h), generator_output]))
        truth_input = Input(shape=(self.mini_batch_size,self.orderLength,1),name='truth_input')
        discriminator_input_truth = Concatenate(axis=2)\
            ([Reshape((1, self.lstm_out_length,1))(lstm_output_h), truth_input])

        #gradient penelty
        averaged_samples = RandomWeightedAverage()([discriminator_input_fake, discriminator_input_truth])

        #discriminator
        D = Sequential(name='discriminator')
        D.add(Conv2D(512,(3,3),padding='same',  input_shape=(self.mini_batch_size, self.orderLength+self.historyLength,1)))
        D.add(Activation('relu'))
        D.add(Conv2D(256, (3,3),padding='same'))
        D.add(Activation('relu'))
        D.add(Conv2D(128,(3,3),padding='same'))
        D.add(Activation('relu'))
        D.add(Flatten())
        D.add(Dense(1))
        self.D = D
        discriminator_output_fake = D(discriminator_input_fake)
        discriminator_output_truth = D(discriminator_input_truth)
        averaged_samples_output = D(averaged_samples)

        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=1)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

        self.gen = Model(inputs=[history_input,noise_input], outputs= generator_output)
        self.model_truth = Model(inputs=[history_input,noise_input, truth_input],\
             outputs= [discriminator_output_fake,discriminator_output_truth,averaged_samples_output])
        self.model_fake = Model(inputs=[history_input,noise_input],\
             outputs= discriminator_output_fake)
        optimizer = Adam(0.0005, beta_1=0.5, beta_2=0.9)
        self.gen.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.gen.summary()
        for layer in self.model_truth.layers:
            layer.trainable = False
        self.model_truth.get_layer(name='discriminator').trainable = True
        self.model_truth.get_layer(name='lstm_critic').trainable = True
        #self.model_truth.get_layer(name='dense_critic').trainable = True
        self.model_truth.compile(optimizer=optimizer, \
            loss=[self.w_loss,self.w_loss,partial_gp_loss])
        for layer in self.model_fake.layers:
            layer.trainable = True
        self.model_fake.get_layer(name='discriminator').trainable = False
        self.model_fake.get_layer(name='lstm_critic').trainable = False
        #self.model_fake.get_layer(name='dense_critic').trainable = False
        self.model_fake.compile(optimizer=optimizer, loss=self.w_loss)
        self.model_fake.summary()
        self.model_truth.summary()

    def fit(self, train_steps=50001, buy_sell_tag=0, batch_size=32, gnr_path='gnr'):
        data = np.load(self.data_path, mmap_mode='r')
        for i in range(train_steps):
            ## gen noise init
            positive_y = np.ones((batch_size, 1), dtype=np.float32)
            negative_y = -positive_y
            dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

            for j in range(10):
                noise = np.random.uniform(-1,1 , size=[batch_size, self.noiseLength])
                ## train/fake init
                idx = np.random.randint(0, data.shape[0])
                orderStreams_train = data[idx].copy()
                orderStreams_train_history = orderStreams_train[:,:self.historyLength,:,0]
                orderStreams_train_truth = orderStreams_train[:,self.historyLength:,:,0:1]
                if(np.min(orderStreams_train_truth[:,:,2,0])==-1):
                    continue
                orderStreams_train_truth[:,:,2:3,0:1] =  ((orderStreams_train_truth[:,:,2:3,0:1]\
                    - orderStreams_train[:,self.historyLength-1:self.historyLength,2:3,0:1]) - 0.001)/0.001
                if(np.min(orderStreams_train_truth[:,:,2,0])>1):
                    continue
                d_loss = self.model_truth.train_on_batch([orderStreams_train_history,noise,\
                    orderStreams_train_truth], [negative_y,positive_y,dummy_y])


            a_loss = self.model_fake.train_on_batch([orderStreams_train_history,noise], positive_y)
            log_mesg = "%d: [D_fake loss: %f,D_truth loss: %f] " % (i, d_loss[0],d_loss[1])
            log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss)
            print(log_mesg)
            if i % 1000 == 0:
               self.gen.save(gnr_path+'_'+str(i))

    def predict(self,save_path='predict_new5_35000_1.npy',length=4000,step_size=1,num_runs=20):
        data = np.load(self.data_path, mmap_mode='r')
        print(data.shape)
        gen = load_model('gnr_new5_35000')

        generated_orders = np.zeros((num_runs, length*step_size+self.historyLength,self.orderLength))
        for j in range(num_runs):
            history = data[4500:4500+self.historyLength,:,0]
            generated_orders[j,:self.historyLength,:] = history
            for i in range(length):
                noise = np.random.uniform(-1,1,size=[1, self.noiseLength])
                orderStreams = gen.predict([np.expand_dims(history,0), noise])\
                    .reshape(step_size, self.orderLength)

                generated_orders[j,self.historyLength+i*step_size:self.historyLength+(i+1)*step_size,:] = orderStreams

                result = [x if x < 1 else 0 for x in 0.001 * orderStreams[:,2] + 0.001]
                result = [x if x< 1 else -1 for x in (result + generated_orders[j,self.historyLength+i*step_size-1,2])]
                generated_orders[j,self.historyLength+i*step_size:self.historyLength+(i+1)*step_size,2] = result
                #history = data[(i+1)*step_size:self.historyLength+(i+1)*step_size,:,0]
                history = generated_orders[j,(i+1)*step_size:self.historyLength+(i+1)*step_size,:]
                if(i % 100 == 0 ):
                    print(str(j)+' runs ' + str(i)+' steps')
        np.save(save_path,generated_orders)
