import pandas as pd
import numpy as np
import os

def read_data_one_day(file_name,start_id,stream_size,batch_size):
    '''
    Input:
    file_name : filename of json_file
    start_id :  start position of orders, index from 0
    stream_size : number of orders in each order stream
    batch_size : number of order streams
    '''
    start = start_id
    end = start_id + stream_size*batch_size - 1

    if(end < start or start < 0):
        return [],[]

    order = pd.read_json('output/'+file_name,orient='records',lines=True)

    time_vector = order['time'].values
    buy_vector = order['buy'].values
    sell_vector = order['sell'].values

    time_index = (time_vector - time_vector[0])/100
    time_index = time_index.astype(int)

    max_t = int(np.max(time_index))
    if(end>max_t):
        return [],[]

    buy_array = np.zeros((600,end - start+1))
    sell_array = np.zeros((600,end - start+1))

    for i in range(len(time_index)):
        if(start <= time_index[i] <= end):
            for price,size in buy_vector[i].items():
                buy_array[int(price)-1,time_index[i]-start] = size
            for price,size in sell_vector[i].items():
                sell_array[int(price)-1,time_index[i]-start] = size
    return buy_array.reshape(-1,stream_size,batch_size,order='F'),sell_array.reshape(-1,stream_size,batch_size,order='F')
