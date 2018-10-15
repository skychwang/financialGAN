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

class orderbook_net(object):
    def __init__(self, historyLength=20, inputLength=9, outputLength=4,batch_size=64,data_path=None):
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.data_path = data_path
        self.batch_size = batch_size
        self.historyLength = historyLength
        self.model = None
        self.build()

    def build(self):
        # build models
        if self.model:
            return self.model
        # lstm cell, to do : attention mechanism
        input_his = Input(shape=(self.historyLength,4))
        lstm_output = LSTM(10)(input_his)
        input_vec = Input(shape=(5,))
        input_all = Concatenate(axis=-1)([input_vec,lstm_output])

        #approxiamate CDA
        G = Sequential(name='net')
        G.add(Dense(128,input_dim=15))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Dense(32))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Dense(8))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Dense(4))
        G.add(Activation('tanh'))
        output_vec = G(input_all)
        self.G = G


        self.net = Model(inputs=[input_vec,input_his], outputs= output_vec)
        optimizer = Adam(0.0001)
        self.net.compile(optimizer=optimizer, loss='mean_squared_error')
        self.net.summary()



    def fit(self, train_steps=300001, batch_size=64, gnr_path='gnr'):
        #self.gen = load_model('gnr_goog12_100000')
        data = np.load(self.data_path, mmap_mode='r')
        for i in range(train_steps):
            idx = np.random.randint(0, data.shape[0])
            orderStreams_train = self.normalize(np.squeeze(data[idx].copy()))
            #print(orderStreams_train.shape)
            #d_history = orderStreams_train[:,:self.historyLength,:]
            #history = (np.squeeze(np.mean((d_history[:,:,0:1] * 10 + d_history[:,:,1:2]),1)) - 11.5)/11.5
            #history_full = self.normalize(orderStreams_train[:,:self.historyLength,8:])
            input_vec = orderStreams_train[:,-2,1:6]
            input_his = orderStreams_train[:,:-1,-4:]
            truth = orderStreams_train[:,-1,6:]
            loss = self.net.train_on_batch([input_vec,input_his],truth,class_weight=[9/20,9/20,1/20,1/20])

            log_mesg = "%d: [loss: %f] " % (i, loss)
            with open('log_goog_order4.txt','a') as f:
                f.write(log_mesg+'\n')
                f.close()
            #print(log_mesg)
            if i % 10000 == 0:
                self.net.save(gnr_path+'_'+str(i))

    def denormalize(self, normArray):
        def denormalize_one_dim(data,maxV=1, minV=0, high=1, low=-1):
            return ((((data - high) * (maxV - minV))/(high - low)) + maxV)

        Array = normArray.copy()
        #New_rep
        maxV =  [16500,1,1,942,150,942,942,3000,3000]
        minV = [0,0,0,916,0,916,916,1,1]
        # GooG
        #maxV =  [205,9,9,9,1,1,942,620]
        #minV = [4,0,0,0,0,0,916,0]
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
        #newArray = np.zeros((Array.shape[0],Array.shape[1],6))
        #newArray[:,:,0] = Array[:,:,0]
        #newArray[:,:,2:] = Array[:,:,4:]
        #newArray[:,:,1] = (Array[:,:,1] * 10 + Array[:,:,2]) * 10 + Array[:,:,3]
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
        # GooG
        maxV =  [23,16500,1,1,942,150,942,942,3000,3000]
        minV = [0,0,0,0,916,0,916,916,1,1]
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
        #if Array.shape[2] == 44:
        for i in range(Array.shape[2]):
            Array[:,:,i] = normalize_one_dim(normArray[:,:,i],maxV=maxV[i],minV=minV[i])
        return Array
        #else:
            # newArray = np.zeros((Array.shape[0],Array.shape[1],8))
            # newArray[:,:,0] = Array[:,:,0]
            # newArray[:,:,4:] = Array[:,:,2:]
            # re = Array[:,:,1].astype(int)
            # newArray[:,:,3] = re % 10
            # re = re // 10
            # newArray[:,:,2] = re % 10
            # newArray[:,:,1] = re // 10
            # for i in range(newArray.shape[2]):
            #     newArray[:,:,i] = normalize_one_dim(newArray[:,:,i],maxV=maxV[i],minV=minV[i])
            # return newArray



    def predict(self,save_path='predict_goog14_new0_4000.npy',length=300000,step_size=1,num_runs=1):

        data = np.load(self.data_path, mmap_mode='r')
        gen = load_model('gnr_goog14_new0_4000')
        #re = np.zeros((num_runs,length*step_size+self.historyLength,2))
        generated_orders = np.zeros((num_runs, length*step_size+self.historyLength,10))
        for j in range(num_runs):
            orderStreams_train = self.normalize(data[0,0:1,:,:,0].copy())
            history = orderStreams_train[:,self.historyLength-1,0:1]
            history_full = orderStreams_train[:,:self.historyLength,1:]
            #history_full = self.normalize(data[0,0:1,:100,6:,0]),1
            #generated_orders[j,:self.historyLength,:] =  self.denormalize(history)
            for i in range(length):
                #noise = np.random.normal(0,0.05,size=[1, self.noiseLength])
                noise_1 = np.random.uniform(-1,1,size=[1, self.noiseLength_1])
                #result = np.zeros((1,1,1))
                #for k in range(1):
                orderStreams = self.denormalize(np.squeeze(gen.predict([history,history_full,noise_1]),-1))
                generated_orders[j,self.historyLength+ i * step_size : self.historyLength+(i+1)*step_size,1:] = orderStreams
                interval = orderStreams[0,:,:1]
                    #re[j,self.historyLength + i * step_size : self.historyLength+(i+1)*step_size,:] = interval
                #result[k,:,0] =  np.round(interval[:,0] * interval[:,1])
                #result_ave = np.mean(result,0)
                r = generated_orders[j:j+1,self.historyLength + i*step_size - 1,0] + interval
                #print(interval)
                generated_orders[j,self.historyLength + i * step_size : self.historyLength+(i+1)*step_size,0] =  r
                history = (np.floor(generated_orders[j:j+1,self.historyLength+ (i+1)*step_size -1,0]/1000000) - 11.5)/11.5
                history_full = self.normalize(generated_orders[j:j+1,(i+1)*step_size : self.historyLength+ (i+1)*step_size,:].copy())[:,:,1:]
                if history > 1:
                    break
                if(i % 1000 == 0 ):
                    print(str(j)+' runs ' + str(i)+' steps')
                    print(interval,history*11.5 + 11.5)
        np.save(save_path,generated_orders)
        #np.save('interval.npy',re)


if __name__ == '__main__':
    net = orderbook_net(data_path='NPY_goog13_new0/080117_1_new.npy')
    net.fit(gnr_path='gnr_goog_order_new6')
