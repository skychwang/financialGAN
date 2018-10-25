import order_vector
import read_json_agg
import lstm_cond_wgan_3 as GAN
import numpy as np

# excel --- json
def get_multiple_days_json():
    print('Start transfer excel to json:')
    order_vector.order_aggregation_multiple_days()
# json --- npy
def  get_multiple_days_npy():
    print('Start transfer json to npy:')
    read_json_agg.read_multiple_days_data(out_dir='GOOG_output/output/'\
        ,out_cancel_dir='GOOG_output/output_cancel/',tgt_dir='NPY_goog12_new/')
    #To do: concatenate with best bid/ask
    read_json_agg.aggregate_multi_days_data(dirPath='NPY_goog12_new/',saveName='agg_data.npy')

#Train GAN
def Train():
    gan_buy = GAN.lstm_cond_gan(data_path='NPY_1/agg_data.npy')
    gan_buy.predict()

if __name__ == '__main__':
    #Excel to json
    get_multiple_days_json()
    #Json to numpy array
    get_multiple_days_npy()
    # TRain GAN
    Train()
