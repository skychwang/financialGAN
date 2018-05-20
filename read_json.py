import pandas as pd
import numpy as np
import os
import sys
import scipy.ndimage as ndimage

def read_one_day_data(out_dir,file_name, out_dir_cancel, file_name_cancel, zero_one=True, history = 100, order_stream=10,batch_size=32):
    '''
    Input:
    file_name : filename of json_file
    '''

    order = pd.read_json(out_dir+file_name,orient='records',lines=True)
    order_cancel = pd.read_json(out_dir_cancel+file_name_cancel,orient='records',lines=True)

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
        max_t = int(np.max(time_index))
        buy_sell_array = np.zeros(((max_t+1),1,1))
        assert buy_vector.shape == sell_vector.shape
        for i in range(len(time_index)):
            if buy_vector[i] or sell_vector[i]:
                buy_sell_array[int(time_index[i]),0,:] = 1
        for i in range(len(time_index_cancel)):
            if buy_vector_cancel[i] or sell_vector_cancel[i]:
                buy_sell_array[int(time_index_cancel[i]),0,:] = 1
    else:
        print(buy_vector_cancel.shape[0])
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


    order_stream = order_stream + history
    num_batches = int(np.floor(buy_sell_array.shape[0]/(order_stream*batch_size)))

    buy_sell_trun = buy_sell_array[0:order_stream*batch_size*num_batches,:]
    if zero_one:
        buy_sell_trun = buy_sell_trun.reshape(num_batches,batch_size,order_stream,1,1,order='C')
    else:
        buy_sell_trun = buy_sell_trun.reshape(num_batches,batch_size,order_stream,240,1,order='C')
    return buy_sell_trun

def read_multiple_days_data():
    raw_orders = [file for file in os.listdir("output/") if file.endswith(".json")]
    raw_orders_cancel = [file for file in os.listdir("output_cancel/") if file.endswith(".json")]
    data = read_one_day_data("output/",raw_orders[0],"output_cancel/",raw_orders_cancel[0])
    for i in range(1,len(raw_orders)):
        data = np.concatenate((data, read_one_day_data("output/",raw_orders[i],'output_cancel/',raw_orders_cancel[i])), axis=0)
    print(data.shape)
    return data

if __name__ == '__main__':
    np.save("data.npy", read_multiple_days_data())
