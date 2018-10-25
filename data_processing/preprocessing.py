import pandas as pd
import numpy as np
import os
import sys
import datetime
import time
import calendar as ca

def read_one_day_data(out_path,out_cancel_path):
    """
    Input:
    out_path: path of submission file
    out_cancel_path: path of cancellation file
    """
    #Some consts:
    minPrice = 6
    maxPrice = 9

    #Read submission and cancellation
    order = pd.read_json(out_path,orient='records',lines=True)
    order_cancel = pd.read_json(out_cancel_path,orient='records',lines=True)

    #Parse current time from filename
    parse_time = out_path.split('/')[2].split("_")[0]
    month = int(parse_time[:2])
    date = int(parse_time[2:4])
    year = 2000 + int(parse_time[4:6])
    time_start = datetime.datetime(year, month, date, 9, 30, 0, 000000)
    time_start = ca.timegm(time_start.timetuple())*1000

    #Extract time, buy(quantity,price), sell(quantity,price) for submissions
    time_vector = order['time'].values
    buy_vector = order['buy'].values
    sell_vector = order['sell'].values
    #Extract time, buy(quantity,price), sell(quantity,price) for cancellation
    time_cancel_vector = order_cancel['time'].values
    buy_cancel_vector = order_cancel['buy'].values
    sell_cancel_vector = order_cancel['sell'].values

    #convert time to relevent time(use time_start as reference)
    time_index = (time_vector - time_start)
    time_index = time_index.astype(int)
    time_index_cancel = (time_cancel_vector - time_start)
    time_index_cancel = time_index_cancel.astype(int)

    #Represent each submission as a vector: [time,type(2-dim),price,quantity]
    buy_sell_array = []
    for i in range(len(buy_vector)):
        time = time_index[i] - 1
        for price,size in buy_vector[i].items():
            buy_sell_array.append([time,0,0,minPrice + float(price)/100 ,size])

        for price,size in sell_vector[i].items():
            buy_sell_array.append([time,1,0,maxPrice + float(price)/100 ,size])
    #Represent each cancellation as a vector: [time,type(2-dim),price,quantity]
    for i in range(len(buy_cancel_vector)):
        time = time_index_cancel[i] - 1
        for price,size in buy_cancel_vector[i].items():
            buy_sell_array.append([time,0,1,minPrice + float(price)/100 ,size])
        for price,size in sell_cancel_vector[i].items():
            buy_sell_array.append([time,1,1,maxPrice + float(price)/100 ,size])

    #Sort orders by time
    data = np.expand_dims(np.array(sorted(buy_sell_array,key=lambda x: (x[0],x[2]) )),-1)

    #Add time slot dimension(1st dim) and change time dimension to be inter-arrival time
    data_new = np.zeros((data.shape[0],6,1))
    data_new[:,2:,0:] = data[:,1:,:]
    for i in range(data.shape[0]):
        #time slot
        data_new[i,0,0] = data[i,0,0] // 10**6
        #inter-arrival
        if i == 0:
            data_new[i,1,0] = data[i,0,0]
        else:
            data_new[i,1,0] = data[i,0,0] - data[i-1,0,0]
    return data_new

def read_multiple_days_data(out_dir,out_cancel_dir,tgt_dir,isTraing=True):
    """
    This function transform muliple days of data into numpy Array
    Input:
    isTraing: if True generated data in training format
    out_dir: dir of submission file
    out_cancel_dir: dir of cancellation file
    tgt_dir: dir to save
    """
    #Read list of files
    raw_orders = [file for file in os.listdir("out_dir") if file.endswith(".json")]


    for i in range(len(raw_orders)):
        raw_path = os.path.join(out_dir+raw_orders[i])
        cancel_path = os.path.join(out_cancel_dir+raw_orders[i])

        #path to save the transformed data
        tgt_path = os.path.join(tgt_dir+raw_orders[i].replace('.json','.npy'))

        #use this to save reshaped form(for training)
        if isTraining:
            np.save(tgt_path, reshape_data(read_one_day_data(raw_path,cancel_path)))
        #use this to save non_reshaped form
        else:
            np.save(tgt_path,read_one_day_data(raw_path,cancel_path))

#Notice: order_size here becomes 10, which is different from the output of
#read_one_day_data, this is because best bid/ask information are added here,
#which is generated in a seperate function.
def reshape_data(buy_sell_array, history=100,order_stream=1,\
        step_size=1,batch_size=64,order_size = 10):
    """
    This function is used to reshape data to be ready for training
    Inputs:
    buy_sell_array: not reshaped data Array
    history: number of orders used as history
    order_stream: number of orders the generator generate
    step_size: minimum distance between two order samples(history + order_stream)
    batch_size: number of orders in one batch
    order_size: dimension of one order
    """
    #Compute the number of samples that can be formed and discard unused orders
    num_samples = int(np.floor((buy_sell_array.shape[0]-history \
        - order_stream + step_size)/(step_size)));
    #Reshape array into shape (num_samples,sample)
    buy_sell_trun = np.zeros((num_samples, order_stream + history,order_size,1))
    for i in range(num_samples):
            buy_sell_trun[i,:,:,:] = \
                buy_sell_array[step_size*i:step_size*i+history+order_stream,:,:]
    #Divide samples into different groups so that samples within
    #one group are independent to each other
    num_groups = int(np.ceil((2 * history + order_stream)/step_size))
    buy_sell = buy_sell_trun[::num_groups]
    for i in range(1,num_groups):
        buy_sell = np.concatenate((buy_sell,buy_sell_trun[i::num_groups]))
    #Compute #batches and discard unused samples
    num_batches = int(np.floor((buy_sell.shape[0])/(batch_size)))
    buy_sell_output = np.zeros((num_batches,batch_size, order_stream + history,order_size,1))
    for i in range(num_batches):
        buy_sell_output[i,:,:,:,:] = buy_sell[i*batch_size:(i+1)*batch_size,:,:,:]

    return buy_sell_output

def aggregate_multi_days_data(dirPath,saveName):
    """
    This function aggregate orders of multiple days
    Inputs:
    dirPath: directory including each day's file
    saveName: name of the saved file
    """
    fileRecognizer = '_1.npy'
    raw_path = [file for file in os.listdir(dirPath) if file.endswith(fileRecognizer)]
    #Sort files in lexicographical order
    #raw_path.sort(key= lambda file:int(file[2:4]))
    #Concatenate all days' data
    raw_data = np.load(dirPath + raw_path[0])
    for i in range(1,len(raw_path)):
        raw_data = np.concatenate((raw_data,np.load(dirPath  + raw_path[i])))
    raw_data = raw_data.astype(float)
    np.save(dirPath+saveName,raw_data)
