import pandas as pd
import numpy as np
import os
import sys

def read_one_day_data(file_name,start_id=0,stream_size=10,batch_size=500):
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
    buy_array = buy_array.reshape(batch_size,stream_size,-1,1,order='F')
    sell_array = sell_array.reshape(batch_size,stream_size,-1,1,order='F')
    return np.concatenate((buy_array, sell_array), axis=3)

def read_multiple_days_data():
    raw_orders = [file for file in os.listdir("output/") if file.endswith(".json")]
    data = read_one_day_data(raw_orders[0])
    for raw_order in raw_orders[1:]:
        data = np.concatenate((data, read_one_day_data(raw_order)), axis=0)
    print(data.shape)
    return data;
        
if __name__ == '__main__':
    np.save("data.npy", read_multiple_days_data())
