import pandas as pd
import numpy as np
import os
import sys
import scipy.ndimage as ndimage
import datetime
import time
import calendar as ca

COUNT = 0

#def convert_time(time):
#    time_new = np.reshape(np.squeeze(time),(101,2))
#    history = time_new[:100,:]
#    order = time_new[100:,:]
    #print(order)
    # converted_time = np.zeros((101,8))
    #
    # def translate_time(time):
    #     transfered_time = []
    #     for _ in range(8):
    #         r = time % 10
    #         transfered_time.insert(0,r)
    #         time = time // 10
    #     return transfered_time
    #
    # for i in range(100):
    #     converted_time[i,:] = translate_time(history[i,1])
    #
    # #ref = np.mean(history,-1)
    # ref = history[99:,1:2]
    # #print(ref)
    # if order.shape[0] == 1:
    #     interval = order[:,1:2] - ref
    # else:
    #     interval = order[:,1:2] - np.concatenate((ref,order[-1,1:2]),-1)
    # for i in range(100,101):
    #     converted_time[i,6] = order[i-100,0]
    #     converted_time[i,7] = interval[i-100,0] / order[i-100,0]
    #     #print(interval)
    #converted_time = np.zeros((101,4))

    #return np.expand_dims(np.expand_dims(converted_time,0),0)

def read_one_day_data(out_path,out_cancel_path):
    '''
    Input:
    out_path: path of excel file
    '''

    order = pd.read_json(out_path,orient='records',lines=True)
    order_cancel = pd.read_json(out_cancel_path,orient='records',lines=True)


    parse_time = out_path.split('/')[2].split("_")[0]
    month = int(parse_time[:2])
    date = int(parse_time[2:4])
    year = 2000 + int(parse_time[4:6])
    time_start = datetime.datetime(year, month, date, 9, 30, 0, 000000)
    time_start = ca.timegm(time_start.timetuple())*1000

    time_vector = order['time'].values
    buy_vector = order['buy'].values
    sell_vector = order['sell'].values
    time_cancel_vector = order_cancel['time'].values
    buy_cancel_vector = order_cancel['buy'].values
    sell_cancel_vector = order_cancel['sell'].values
    print(len(buy_vector),len(buy_cancel_vector))

    #convert time to index of numpy array
    time_index = (time_vector - time_start)
    time_index = time_index.astype(int)
    time_index_cancel = (time_cancel_vector - time_start)
    time_index_cancel = time_index_cancel.astype(int)

    #Q-GAN
    buy_sell_array = []
    for i in range(len(buy_vector)):
        #if time_index[i] > 0 and time_index[i] < 23400000:
            #time = time_index[i] + np.random.normal(0,0.1,1)[0]
        time = time_index[i] - 1
        for price,size in buy_vector[i].items():
            #if(size <= 500):
            #size = size + np.random.normal(0,0.08*size,1)[0]
            #if(size<0):
            #    size = 1
            buy_sell_array.append([time,0,0,6 + float(price)/100 ,size])

        for price,size in sell_vector[i].items():
            #if(size <= 500):
            #size = size + np.random.normal(0,0.08*size,1)[0]
            #if(size<0):
            #    size = 1
            buy_sell_array.append([time,1,0,9 + float(price)/100 ,size])

    for i in range(len(buy_cancel_vector)):
        #if time_index_cancel[i] > 0 and time_index_cancel[i] < 23400000:
            #time = time_index_cancel[i] + np.random.normal(0,0.1,1)[0]
        time = time_index_cancel[i] - 1
        for price,size in buy_cancel_vector[i].items():
            #if(size <= 500):
            #size = size + np.random.normal(0,0.08*size,1)[0]
            #if(size<0):
            #    size = 1
            buy_sell_array.append([time,0,1,6 + float(price)/100 ,size])
        for price,size in sell_cancel_vector[i].items():
            #if(size <= 500):
            #size = size + np.random.normal(0,0.08*size,1)[0]
            #if(size<0):
            #    size = 1
            buy_sell_array.append([time,1,1,9 + float(price)/100 ,size])

    data = np.expand_dims(np.array(sorted(buy_sell_array,key=lambda x: (x[0],x[2]) )),-1)

    # data_new = np.zeros((data.shape[0],6,1))
    # data_new[:,1:,0:] = data[:,:,:]
    # for i in range(data.shape[0]):
    #     if i <= 1:
    #         data_new[i ,0,0] = data_new[i,1,0]
    #     elif i < 2000:
    #         data_new[i ,0,0] = np.mean(data[1:i,0,0] - data[0:i-1,0,0])
    #     else:
    #         data_new[i ,0,0] = np.mean(data[i-1999:i,0,0] - data[i-2000:i-1,0,0])

    data_new = np.zeros((data.shape[0],6,1))
    data_new[:,2:,0:] = data[:,1:,:]
    for i in range(data.shape[0]):
        data_new[i,0,0] = data[i,0,0] // 10**6
        if i == 0:
            data_new[i,1,0] = data[i,0,0]
        else:
            data_new[i,1,0] = data[i,0,0] - data[i-1,0,0]
    print(data_new.shape)
    return data_new

def read_multiple_days_data():
    raw_orders = [file for file in os.listdir("GOOG_output/output/") if file.endswith(".json")]

    for i in range(len(raw_orders)):
        raw_path = os.path.join('GOOG_output/output/'+raw_orders[i])
        cancel_path = os.path.join('GOOG_output/output_cancel/'+raw_orders[i])
        print(raw_path,cancel_path)

        tgt_path = os.path.join('NPY_goog11_new/'+raw_orders[i].replace('.json','.npy'))
        #np.save(tgt_path, np.expand_dims(np.squeeze(read_one_day_data(raw_path,cancel_path)),0))
        #np.save(tgt_path, reshape_data(read_one_day_data(raw_path,cancel_path)))
        np.save(tgt_path,read_one_day_data(raw_path,cancel_path))

def Get_interval(data,ref):
    high = np.squeeze(data)
    reference = np.squeeze(data)
    low = np.concatenate((reference,hight[:-1,:]),0)
    interval =  high - low


def reshape_data(buy_sell_array, history=100,order_stream=1,step_size=1,batch_size=64):
    num_samples = int(np.floor((buy_sell_array.shape[0]-history - order_stream + step_size)/(step_size)));

    buy_sell_trun = np.zeros((num_samples, order_stream + history,10,1))

    for i in range(num_samples):
            buy_sell_trun[i,:,:,:] = buy_sell_array[step_size*i:step_size*i+history+order_stream,:,:]

    num_groups = int(np.ceil((2 * history + order_stream)/step_size))
    buy_sell = buy_sell_trun[::num_groups]
    for i in range(1,num_groups):
        buy_sell = np.concatenate((buy_sell,buy_sell_trun[i::num_groups]))

    num_batches = int(np.floor((buy_sell.shape[0])/(batch_size)))

    buy_sell_output = np.zeros((num_batches,batch_size, order_stream + history,10,1))

    for i in range(num_batches):
        buy_sell_output[i,:,:,:,:] = buy_sell[i*batch_size:(i+1)*batch_size,:,:,:]

    #buy_sell = np.zeros((num_batches,batch_size, order_stream + history,52,1))
    #for i in range(num_batches):
    #    for j in range(batch_size):
    #        buy_sell[i,j,:,8:,0] = buy_sell_output[i,j,:,2:,0]
    #        buy_sell[i,j,:,:8,0] = convert_time(buy_sell_output[i,j,:,:2,0])
    print(buy_sell_output.shape)


    return buy_sell_output

def normalize(array, maxV, minV, high=1, low=-1):
    return (high - (((high - low) * (maxV - array)) / (maxV - minV)))

def aggregate_multi_days_data():
    raw_path = [file for file in os.listdir("NPY_goog12_new/") if file.endswith("_1.npy")]
    raw_path.sort(key= lambda file:int(file[2:4]))
    print(raw_path)
    raw_data = np.load("NPY_goog12_new/" + raw_path[0])
    for i in range(1,len(raw_path)):
        raw_data = np.concatenate((raw_data,np.load("NPY_goog12_new/" + raw_path[i])))

    raw_data = raw_data.astype(float)
    np.save('NPY_goog12_new/all_data.npy',raw_data)
    #print(COUNT)
    #raw_data = reshape_data(raw_data)
    #print(raw_data.shape)
    print((np.max(raw_data[:,:,:,0,0])))
    print((np.max(raw_data[:,:,:,1,0])))
    print((np.max(raw_data[:,:,:,2,0])))
    print((np.max(raw_data[:,:,:,3,0])))
    print((np.max(raw_data[:,:,:,4,0])))
    print((np.max(raw_data[:,:,:,5,0])))
    print((np.max(raw_data[:,:,:,6,0])))
    print((np.max(raw_data[:,:,:,7,0])))
    print((np.max(raw_data[:,:,:,8,0])))
    print((np.max(raw_data[:,:,:,9,0])))
    print((np.max(raw_data[:,:,:,10,0])))
    print((np.max(raw_data[:,:,:,11,0])))
    print((np.min(raw_data[:,:,:,0,0])))
    print((np.min(raw_data[:,:,:,1,0])))
    print((np.min(raw_data[:,:,:,2,0])))
    print((np.min(raw_data[:,:,:,3,0])))
    print((np.min(raw_data[:,:,:,4,0])))
    print((np.min(raw_data[:,:,:,5,0])))
    print((np.min(raw_data[:,:,:,6,0])))
    print((np.min(raw_data[:,:,:,7,0])))
    print((np.min(raw_data[:,:,:,8,0])))
    print((np.min(raw_data[:,:,:,9,0])))
    print((np.min(raw_data[:,:,:,10,0])))
    print((np.min(raw_data[:,:,:,11,0])))
    #raw_data[:,:,:,0,:] = normalize(raw_data[:,:,:,0,:],maxV=24,minV=0)
    #raw_data[:,:,:,1,:] = normalize(raw_data[:,:,:,1,:],maxV=60,minV=0)
    #raw_data[:,:,:,2,:] = normalize(raw_data[:,:,:,2,:],maxV=60,minV=0)
    #raw_data[:,:,:,3,:] = normalize(raw_data[:,:,:,3,:],maxV=10,minV=0)
    #raw_data[:,:,:,4,:] = normalize(raw_data[:,:,:,4,:],maxV=np.max(raw_data[:,:,:,4,0]),minV=np.min(raw_data[:,:,:,4,0]))
    #raw_data[:,:,:,5,:] = normalize(raw_data[:,:,:,5,:],maxV=np.max(raw_data[:,:,:,5,0]),minV=np.min(raw_data[:,:,:,5,0]))
    #raw_data[:,:,:,6,:] = normalize(raw_data[:,:,:,6,:],maxV=np.max(raw_data[:,:,:,6,0]),minV=np.min(raw_data[:,:,:,6,0]))
    #raw_data[:,:,:,7,:] = normalize(raw_data[:,:,:,7,:],maxV=np.max(raw_data[:,:,:,7,0]),minV=np.min(raw_data[:,:,:,7,0]))
    #Generate NPY(reshaped)
    #np.save("NPY_goog11_new0/agg_data.npy",raw_data)
