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
    #read_json_agg.read_multiple_days_data()
    read_json_agg.aggregate_multi_days_data()

#Train Q-GAN
def Q_GAN_Train():
    gan_buy = GAN.lstm_cond_gan(data_path='NPY_1/agg_data.npy')
    #gan_buy.fit(gnr_path='gnr_buy',buy_sell_tag=0)
    #gan_sell = GAN.lstm_cond_gan(data_path='NPY/agg_data.npy')
    #gan_sell.fit(gnr_path='gnr_sell',buy_sell_tag=1)
    #gan_cancel_buy = GAN.lstm_cond_gan(data_path='NPY/agg_data.npy')
    #gan_cancel_buy.fit(gnr_path='gnr_cancel_buy',buy_sell_tag=2)
    #gan_cancel_sell = GAN.lstm_cond_gan(data_path='NPY/agg_data.npy')
    #gan_cancel_sell.fit(gnr_path='gnr_cancel_sell',buy_sell_tag=3)
    gan_buy.predict()

#Train Z_O_GAN
def Z_O_GAN_Train():
    #gan_buy = GAN.lstm_cond_gan_01(data_path='NPY/080116_100.npy',data_cancel_path='NPY_cancel/080116_100.npy')
    #gan_buy.fit(gnr_path='gnr_buy',buy_sell_tag=0)
    gan_sell = GAN.lstm_cond_gan_01(data_path='NPY/080116_100.npy',data_cancel_path='NPY_cancel/080116_100.npy')
    gan_sell.fit(gnr_path='gnr_sell',buy_sell_tag=1)
    #gan_cancel_buy = GAN.lstm_cond_gan_01(data_path='NPY/080116_100.npy',data_cancel_path='NPY_cancel/080116_100.npy')
    #gan_cancel_buy.fit(gnr_path='gnr_cancel_buy',buy_sell_tag=2)
    #gan_cancel_sell = GAN.lstm_cond_gan_01(data_path='NPY/080116_100.npy',data_cancel_path='NPY_cancel/080116_100.npy')
    #gan_cancel_sell.fit(gnr_path='gnr_cancel_sell',buy_sell_tag=3)
    #gan_buy.predict()


if __name__ == '__main__':
    #get_multiple_days_json()
    get_multiple_days_npy()
    # Train Zero_one GAN
    #Z_O_GAN_Train()
    # TRain Quantity GAN
    Q_GAN_Train()
