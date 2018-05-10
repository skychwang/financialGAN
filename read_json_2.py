import pandas as pd
import numpy as np
import os
import sys
import scipy.ndimage as ndimage

def read_one_day_data(file_name, history = 100, order_stream=100,batch_size=256):
    '''
    Input:
    file_name : filename of json_file
    '''

    order = pd.read_json('output/'+file_name,orient='records',lines=True)

    time_vector = order['time'].values
    buy_vector = order['buy'].values
    sell_vector = order['sell'].values

    time_index = (time_vector - time_vector[0])/100
    time_index = time_index.astype(int)

    max_t = int(np.max(time_index))

    buy_sell_array = np.zeros(((max_t+1),1,2))
    order_stream = order_stream + history

    for i in range(len(time_index)):
        if buy_vector[i]:
            buy_sell_array[int(time_index[i]),0,:] = 1

    num_batches = int(np.floor(buy_sell_array.shape[0]/(order_stream*batch_size)))

    buy_sell_trun = buy_sell_array[0:order_stream*batch_size*num_batches,:]
    buy_sell_trun = buy_sell_trun.reshape(num_batches,batch_size,order_stream,1,2,order='C')
    return buy_sell_trun

def read_multiple_days_data():
    raw_orders = [file for file in os.listdir("output/") if file.endswith(".json")]
    data = read_one_day_data(raw_orders[0])
    for raw_order in raw_orders[1:]:
        data = np.concatenate((data, read_one_day_data(raw_order)), axis=0)
    print(data.shape)
    #data[data>800]=0
    return data

if __name__ == '__main__':
    np.save("data_4.npy", read_multiple_days_data())