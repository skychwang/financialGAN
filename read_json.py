import pandas as pd
import numpy as np
import os
import sys
import scipy.ndimage as ndimage

def read_one_day_data(out_path, zero_one=False, history = 100, order_stream=50,step_size=50, batch_size=32):
    '''
    Input:
    out_path: path of excel file
    zero_one: if this is to generate 0-1 data
    hisroty: the length of history
    order_stream: the length of order_stream
    step_size: sliding window step size
    batch_size: batch size of each batch
    '''

    order = pd.read_json(out_path,orient='records',lines=True)

    time_vector = order['time'].values
    buy_vector = order['buy'].values
    sell_vector = order['sell'].values

    #convert time to index of numpy array
    time_index = (time_vector - time_vector[0])/100
    time_index = time_index.astype(int)

    #zero-one GAN
    if zero_one:
        #time range is defined by max_t
        max_t = int(np.max(time_index))
        buy_sell_array = np.zeros(((max_t+1),2,1))
        #fill in ones
        for i in range(len(time_index)):
            if buy_vector[i]:
                buy_sell_array[int(time_index[i]),0,:] = 1
            if sell_vector[i]:
                buy_sell_array[int(time_index[i]),1,:] = 1

    else:
        max_t = buy_vector.shape[0]
        buy_sell_array = np.zeros(((max_t),1200,1))
        for i in range(max_t):
            for price,size in buy_vector[i].items():
                buy_sell_array[i,int(int(price))-1,0] += size
            for price,size in sell_vector[i].items():
                buy_sell_array[i,int(int(price))-1+600,0] += size

    # Reshape(implement sliding window here)
    # Compute Number of Batches
    num_batches = int(np.floor((buy_sell_array.shape[0]-history - order_stream + step_size)/(step_size*batch_size)))

    if zero_one:
        buy_sell_trun = np.zeros((num_batches,batch_size, order_stream + history,2,1))
    else:
        buy_sell_trun = np.zeros((num_batches,batch_size, order_stream + history,1200,1))

    for i in range(num_batches):
        for j in range(batch_size):
            #the length of one block is order_stream + history(this is where we put history in)
            buy_sell_trun[i,j,:,:,:] = buy_sell_array[step_size*(i*batch_size+j):step_size*(i*batch_size+j)+ order_stream + history,:,:]
    print(buy_sell_array.shape)
    return buy_sell_trun

def read_multiple_days_data():
    raw_orders = [file for file in os.listdir("output/") if file.endswith(".json")]
    raw_orders_cancel = [file for file in os.listdir("output_cancel/") if file.endswith(".json")]
    for i in range(len(raw_orders)):
        #we put cancel orders into seperate files
        raw_path = os.path.join('output/'+raw_orders[i])
        cancel_path = os.path.join('output_cancel/'+raw_orders_cancel[i])
        tgt_path = os.path.join('NPY/'+raw_orders[i].replace('.json','.npy'))
        tgt_path_cancel = os.path.join('NPY_cancel/'+raw_orders_cancel[i].replace('.json','.npy'))
        np.save(tgt_path, read_one_day_data(raw_path))
        np.save(tgt_path_cancel, read_one_day_data(cancel_path))

if __name__ == '__main__':
    read_multiple_days_data()
