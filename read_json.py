import pandas as pd
import numpy as np
import os
import sys
import scipy.ndimage as ndimage

def read_one_day_data(out_path, out_cancel_path, zero_one=True, history = 100, order_stream=10, step_size=1, batch_size=32):
    '''
    Input:
    file_name : filename of json_file
    '''

    order = pd.read_json(out_path,orient='records',lines=True)
    order_cancel = pd.read_json(out_cancel_path,orient='records',lines=True)

    time_vector = order['time'].values
    time_vector_cancel = order_cancel['time'].values
    buy_vector = order['buy'].values
    buy_vector_cancel = order_cancel['buy'].values
    sell_vector = order['sell'].values
    sell_vector_cancel = order['sell'].values

    time_index = (time_vector - time_vector[0])/100
    time_index = time_index.astype(int)
    time_index_cancel = (time_vector_cancel - time_vector_cancel[0])/100
    time_index_cancel = time_index_cancel.astype(int)

    if zero_one:
        #Assume number of orders >= number of cancel orders
        max_t = int(np.max(time_index))
        assert buy_vector.shape[0] >= buy_vector_cancel.shape[0]
        buy_sell_array = np.zeros(((max_t+1),4,1))

        for i in range(len(time_index)):
            if buy_vector[i]:
                buy_sell_array[int(time_index[i]),0,:] = 1
            if sell_vector[i]:
                buy_sell_array[int(time_index[i]),1,:] = 1

        for i in range(len(time_index_cancel)):
            if buy_vector_cancel[i]:
                buy_sell_array[int(time_index_cancel[i]),2,:] = 1
            if buy_vector_cancel[i]:
                buy_sell_array[int(time_index_cancel[i]),3,:] = 1
    else:
        #Assume number of orders >= number of cancel orders
        max_t = buy_vector.shape[0]
        buy_sell_array = np.zeros(((max_t),240,1))
        for i in range(max_t):
            for price,size in buy_vector[i].items():
                buy_sell_array[i,int(int(price))-1,0] += size
            for price,size in sell_vector[i].items():
                buy_sell_array[i,int(int(price))-1+60,0] += size

        max_t_cancel = buy_vector_cancel.shape[0]
        for i in range(max_t_cancel):
            for price,size in buy_vector_cancel[i].items():
                buy_sell_array[i,int(int(price))-1+120,0] += size
            for price,size in sell_vector_cancel[i].items():
                buy_sell_array[i,int(int(price))-1+180,0] += size

    # Reshape(implement sliding window here)
    num_batches = int(np.floor((buy_sell_array.shape[0]-history)/(step_size*batch_size)))

    if zero_one:
        buy_sell_trun = np.zeros((num_batches,batch_size, step_size + history,4,1))
    else:
        buy_sell_trun = np.zeros((num_batches,batch_size, step_size + history,240,1))

    for i in range(num_batches):
        for j in range(batch_size):
            buy_sell_trun[i,j,:,:,:] = buy_sell_array[step_size*(i*batch_size+j):step_size*(i*batch_size+j+1)+ history,:,:]
    return buy_sell_trun

def read_multiple_days_data():
    raw_orders = [file for file in os.listdir("output/") if file.endswith(".json")]
    raw_orders_cancel = [file for file in os.listdir("output_cancel/") if file.endswith(".json")]
    for i in range(len(raw_orders)):
        raw_path = os.path.join('output/'+raw_orders[i])
        cancel_path = os.path.join('output_cancel/'+raw_orders_cancel[i])
        tgt_path = os.path.join('NPY/'+raw_orders[i].replace('.json','.npy'))
        data = np.save(tgt_path, read_one_day_data(raw_path,cancel_path))

if __name__ == '__main__':
    read_multiple_days_data()
