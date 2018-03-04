import pandas as pd
import numpy as np
import os

def detect_cancel(value):
    """
    This function deal with each order, only the last submission
    is valid, all the submission before it is cancelled.
    """
    length = value.shape[0] #the length of the group
    if(value.iloc[length-1]['SIZE'] > 0):
        value_new = value.iloc[0:length-1] # get rid of the last row if it is valid
    else:
        value_new = value
    cancel_df = value_new.loc[value_new.loc[:,'SIZE']>0,:]
    return cancel_df

def get_cancel_order(order_filename):
    """
    This function generates execl file for canelletion orders with raw orders
    Input:
    dir : tdir of the raw order
    file_name : file_name of the raw order
    """

    src_path = os.path.join('RMD/'+order_filename)
    tgt_path = os.path.join('RMD/'+order_filename.replace('Raw','Cancel'))
    example = pd.read_excel(src_path).drop(columns=['Index'])
    example_cancellation = example.groupby(['ORDER_ID'],as_index=False)
    cancellation_trade = example_cancellation.apply(detect_cancel) \
                .reset_index().drop(columns=['level_0','level_1'])
    cancellation_trade.to_excel(tgt_path,index=False)


if __name__ == '__main__':
    get_cancel_order('PN_Order_Raw_080116.xlsx')
