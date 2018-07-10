import pandas as pd
import numpy as np
import os
import sys
import scipy.ndimage as ndimage
import datetime
import time

def read_one_day_data(out_path, out_cancel_path,zero_one=True):
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
    order_cancel = pd.read_json(out_cancel_path,orient='records',lines=True)


    parse_time = out_path.split('/')[1].split("_")[0]
    month = int(parse_time[:2])
    date = int(parse_time[2:4])
    year = 2000 + int(parse_time[4:6])
    time_start = datetime.datetime(year, month, date, 9, 50, 0, 000000)
    time_start = time.mktime(time_start.timetuple())*1000

    time_vector = order['time'].values
    buy_vector = order['buy'].values
    sell_vector = order['sell'].values
    time_cancel_vector = order_cancel['time'].values
    buy_cancel_vector = order_cancel['buy'].values
    sell_cancel_vector = order_cancel['sell'].values

    #convert time to index of numpy array
    time_index = (time_vector - time_start)/100
    time_index = time_index.astype(int)
    time_index_cancel = (time_cancel_vector - time_start)/100
    time_index_cancel = time_index_cancel.astype(int)

    #T-GAN
    if zero_one:
        #time range is defined by max_t
        max_t = 210000
        buy_sell_array = np.zeros(((max_t),4,1))
        #fill in ones
        for i in range(len(buy_vector)):
            if buy_vector[i] and time_index[i] > 0 and time_index[i] < max_t:
                buy_sell_array[int(time_index[i]),0,:] = 1
            if sell_vector[i] and time_index[i] > 0 and time_index[i] < max_t:
                buy_sell_array[int(time_index[i]),1,:] = 1

        for i in range(len(buy_cancel_vector)):
            if buy_cancel_vector[i] and time_index_cancel[i] > 0 and time_index_cancel[i] < max_t:
                buy_sell_array[int(time_index_cancel[i]),2,:] = 1
            if sell_cancel_vector[i] and time_index_cancel[i] > 0 and time_index_cancel[i] < max_t:
                buy_sell_array[int(time_index_cancel[i]),3,:] = 1
    #Q-GAN
    else:
        max_t = 210000
        buy_sell_array = np.zeros((max_t,2400,1))
        for i in range(len(buy_vector)):
            for price,size in buy_vector[i].items():
                buy_sell_array[i,int(int(price))-1,0] += size
            for price,size in sell_vector[i].items():
                buy_sell_array[i,int(int(price))-1+600,0] += size

        for i in range(len(buy_cancel_vector)):
            for price,size in buy_cancel_vector[i].items():
                buy_sell_array[i,int(int(price))-1+1200,0] += size
            for price,size in sell_cancel_vector[i].items():
                buy_sell_array[i,int(int(price))-1+1800,0] += size

        zero_one_tag = np.sum(buy_sell_array[:,:,0],axis=1) > 0
        zero_one_sum = np.sum(zero_one_tag)

        buy_sell_trun = np.zeros((zero_one_sum,2400,1))
        j = 0
        for i in range(max_t):
            if(zero_one_tag[i] > 0):
                buy_sell_trun[j] = buy_sell_array[i]
                j = j + 1
        assert j == zero_one_sum
        buy_sell_array = buy_sell_trun

    print(buy_sell_array.shape)
    return buy_sell_array

def read_multiple_days_data():
    raw_orders = [file for file in os.listdir("output/") if file.endswith(".json")]
    raw_orders_cancel = [file for file in os.listdir("output_cancel/") if file.endswith(".json")]

    for i in range(len(raw_orders)):
        raw_path = os.path.join('output/'+raw_orders[i])
        cancel_path = os.path.join('output_cancel/'+raw_orders_cancel[i])

        tgt_path = os.path.join('NPY/'+raw_orders[i].replace('.json','.npy'))
        np.save(tgt_path, read_one_day_data(raw_path,cancel_path))
        #np.save(tgt_path, reshape_data(read_one_day_data(raw_path,cancel_path)))

def reshape_data(buy_sell_array, zero_one=True, history=100,order_stream=1,step_size=1,batch_size=32):
    num_samples = int(np.floor((buy_sell_array.shape[0]-history - order_stream + step_size)/(step_size)));
    if zero_one:
        buy_sell_trun = np.zeros((num_samples, order_stream + history,4,1))
    else:
        buy_sell_trun = np.zeros((num_samples, order_stream + history,2400,1))

    for i in range(num_samples):
            buy_sell_trun[i,:,:,:] = buy_sell_array[step_size*i:step_size*i+history+order_stream,:,:]

    num_groups = int(np.ceil((2 * history + order_stream)/step_size))
    buy_sell = buy_sell_trun[::num_groups]
    for i in range(1,num_groups):
        buy_sell = np.concatenate((buy_sell,buy_sell_trun[i::num_groups]))

    num_batches = int(np.floor((buy_sell.shape[0])/(batch_size)))

    if zero_one:
        buy_sell_output = np.zeros((num_batches,batch_size, order_stream + history,4,1))
    else:
        buy_sell_output = np.zeros((num_batches,batch_size, order_stream + history,2400,1))

    for i in range(num_batches):
        #for j in range(batch_size):
            #the length of one block is order_stream + history(this is where we put history in)
        buy_sell_output[i,:,:,:,:] = buy_sell[i*batch_size:(i+1)*batch_size,:,:,:]
    print(buy_sell_output.shape)


    return buy_sell_output



def aggregate_multi_days_data(zero_one=True, history = 100, order_stream=1,step_size=1, batch_size=32):
    raw_path = [file for file in os.listdir("NPY/") if file.endswith("_100.npy")]
    raw_data = np.load("NPY/" + raw_path[0])
    for i in range(1,len(raw_path)):
        raw_data = np.concatenate((raw_data,np.load("NPY/" + raw_path[i])))

    #Generate NPY(reshaped)
    np.save("NPY/agg_data.npy",reshape_data(raw_data))

    #Generate NPY_1(non-reshaped)
    #np.save('NPY_1/agg_data.npy',raw_data)


if __name__ == '__main__':
    read_multiple_days_data()
