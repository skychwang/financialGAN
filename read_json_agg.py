import pandas as pd
import numpy as np
import os
import sys
import scipy.ndimage as ndimage

def read_one_day_data(out_path, zero_one=False, history = 100, order_stream=1,step_size=1, batch_size=32):
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
    print(buy_sell_array.shape)
    return buy_sell_array

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

def reshape_data(buy_sell_array, zero_one=False, history=100,order_stream=1,step_size=1,batch_size=32):
    num_samples = int(np.floor((buy_sell_array.shape[0]-history - order_stream + step_size)/(step_size)));
    if zero_one:
        buy_sell_trun = np.zeros((num_samples, order_stream + history,2,1))
    else:
        buy_sell_trun = np.zeros((num_samples, order_stream + history,1200,1))

    for i in range(num_samples):
            buy_sell_trun[i,:,:,:] = buy_sell_array[step_size*i:step_size*i+history+order_stream,:,:]

    num_groups = int(np.ceil((2 * history + order_stream)/step_size))
    buy_sell = buy_sell_trun[::num_groups]
    for i in range(1,num_groups):
        buy_sell = np.concatenate((buy_sell,buy_sell_trun[i::num_groups]))

    num_batches = int(np.floor((buy_sell.shape[0])/(batch_size)))

    if zero_one:
        buy_sell_output = np.zeros((num_batches,batch_size, order_stream + history,2,1))
    else:
        buy_sell_output = np.zeros((num_batches,batch_size, order_stream + history,1200,1))

    for i in range(num_batches):
        #for j in range(batch_size):
            #the length of one block is order_stream + history(this is where we put history in)
        buy_sell_output[i,:,:,:,:] = buy_sell[i*batch_size:(i+1)*batch_size,:,:,:]
    print(buy_sell_output.shape)


    return buy_sell_output



def aggregate_multi_days_data(zero_one=False, history = 100, order_stream=1,step_size=1, batch_size=32):
    raw_path = [file for file in os.listdir("NPY/") if file.endswith(".npy")]
    cancel_path = [file for file in os.listdir("NPY_cancel/") if file.endswith(".npy")]
    raw_data = np.load("NPY/" + raw_path[0])
    cancel_data = np.load("NPY_cancel/" + cancel_path[0])
    for i in range(1,len(raw_path)):
        raw_data = np.concatenate((raw_data,np.load("NPY/" + raw_path[i])))
        cancel_data = np.concatenate((cancel_data,np.load("NPY_cancel/" + cancel_path[i])))

    np.save("NPY/agg_data.npy",reshape_data(raw_data))
    np.save("NPY_cancel/agg_data_cancel.npy",reshape_data(cancel_data))


if __name__ == '__main__':
    read_multiple_days_data()
