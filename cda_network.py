import time

from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers.merge import _Merge
from keras.optimizers import RMSprop, Adam,Nadam
from keras.callbacks import ModelCheckpoint

from read_json import *
from order_vector import *
from functools import partial
from discrimination import *

class orderbook_net(object):
    def __init__(self):
        self.net = None
        self.build()

    def build(self):
        # build models
        if self.net:
            return self.net

        input_his = Input(shape=(8,))

        G = Sequential(name='discriminator')
        G.add(Dense(256*3,input_dim=8))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Reshape((16, 16, 3)))
        G.add(Conv2D(128,(3,3),padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Conv2D(64, (3,3),padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Conv2D(32,(3,3),padding='same'))
        G.add(BatchNormalization())
        G.add(Activation('relu'))
        G.add(Flatten())
        G.add(Dense(4))
        output_vec = G(input_his)

        self.net = Model(inputs=input_his, outputs=output_vec)
        optimizer = Adam(0.0001)
        self.net.compile(optimizer=optimizer, loss='mean_squared_error')
        self.net.summary()



    def fit(self, epoch = 600, batch_size=64, gnr_path='gnr'):
        X_train = self.normalize(np.load('train_set_pn.npy'))
        y_train = self.normalize(np.load('train_label_pn.npy'))
        X_test = self.normalize(np.load('test_set_pn.npy'))
        y_test = self.normalize(np.load('test_label_pn.npy'))
        ckpt = ModelCheckpoint(gnr_path, verbose=1, save_best_only=True)

        #Set different weights for price and quantity
        weights = 2*[9/20]
        weights.extend(2*[1/20])

        self.net.fit(X_train,y_train,batch_size=64,epochs= epoch,verbose=2,\
            validation_data=(X_test,y_test),callbacks=[ckpt], class_weight=weights)

    def denormalize(self, normArray):
        def denormalize_one_dim(data,maxV=1, minV=0, high=1, low=-1):
            return ((((data - high) * (maxV - minV))/(high - low)) + maxV)

        Array = normArray.copy()
        #New_rep
        #maxV =  [1,1,942,150,942,942,3000,3000]
        #minV = [0,0,916,0,916,916,1,1]
        #
        #maxV =  [1,1,1,1,1,1,2,2]
        #minV = [0,0,-1,-1,-1,-1,1,1]
        #
        maxV =  [1,1,13,300,13,13,40000,40000]
        minV = [0,0,6,0,6,6,1,1]
        if Array.shape[1] == 8:
            for i in range(4):
                Array[:,i] = denormalize_one_dim(normArray[:,i],maxV=maxV[i],minV=minV[i])
            start = 4
        else:
            start = 0
        for i in range(start,start + 2):
            Array[:,i] = denormalize_one_dim(normArray[:,i],maxV=13,minV=6)
        for i in range(start + 2,start + 4):
            Array[:,i] = denormalize_one_dim(normArray[:,i],maxV=40000,minV=1)
        return Array

    def normalize(self, normArray):
        def normalize_one_dim(array, maxV=1, minV=0, high=1, low=-1):
            return (high - (((high - low) * (maxV - array)) / (maxV - minV)))

        Array = normArray.copy()
        #New_rep
        #maxV =  [1,1,942,150,942,942,3000,3000]
        #minV = [0,0,916,0,916,916,1,1]
        #
        #maxV =  [1,1,1,1,1,1,2,2]
        #minV = [0,0,-1,-1,-1,-1,1,1]
        #
        maxV =  [1,1,13,300,13,13,40000,40000]
        minV = [0,0,6,0,6,6,1,1]
        if Array.shape[1] == 8:
            for i in range(4):
                Array[:,i] = normalize_one_dim(normArray[:,i],maxV=maxV[i],minV=minV[i])
            start = 4
        else:
            start = 0
        for i in range(start,start + 2):
            Array[:,i] = normalize_one_dim(normArray[:,i],maxV=13,minV=6)
        for i in range(start + 2,start + 4):
            Array[:,i] = normalize_one_dim(normArray[:,i],maxV=40000,minV=1)

        return Array

    def predict(self):
        Orders = np.load('real_pn.npy')
        Net = load_model('gnr_goog_pp_pn')

        new_orders = Orders.copy()
        for i in range(1,15000):
            if i % 1000 == 0:
                print(str(i)+'steps\n')
            new_orders[i,4:] = self.denormalize(Net.predict(self.normalize(new_orders[i-1:i,:])))
        np.save('new_gnr_pn',new_orders)

if __name__ == '__main__':
    #net = orderbook_net(data_path='NPY_goog13_new0/080117_1_new1.npy')
    net = orderbook_net()
    #net.fit(gnr_path='gnr_goog_pp_pn')
    net.predict()
