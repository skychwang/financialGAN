import numpy as np

#import order_vector
#import read_json_agg
#import add_orderbook
import train_no_cda as GAN
import train_no_best as GAN_1
import train_no_time as GAN_2
import train_no_cda_new as GAN_3
import train_dcgan as GAN_4
import train_para as GAN_5
import train_sn as GAN_6

# excel --- json
# def get_multiple_days_json():
#     print('Start transfer excel to json:')
#     order_vector.order_aggregation_multiple_days()
# # json --- npy
# def  get_multiple_days_npy():
#     print('Start transfer json to npy:')
#     read_json_agg.read_multiple_days_data(out_dir='GOOG_output/output/'\
#         ,out_cancel_dir='GOOG_output/output_cancel/',tgt_dir='NPY_goog12_new/')
#     add_orderbook.get_cda_data('NPY_goog12_new/')
#     read_json_agg.aggregate_multi_days_data(dirPath='NPY_goog12_new/',saveName='agg_data.npy')

#Train GAN
def Train():
    gan = GAN_2.lstm_cond_gan(data_path='SYN/syn_data_train.npy')
    gan.fit(gnr_path='SYN/generator/gnr_sn_no_time')
    #gan.predict()

if __name__ == '__main__':
    #Excel to json
    #get_multiple_days_json()
    #Json to numpy array
    #get_multiple_days_npy()
    # TRain GAN
    Train()
