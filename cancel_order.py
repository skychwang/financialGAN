import pandas as pd
import numpy as np
import os

def detect_cancel(value):
    """
    Find orders that are cancelled or trasacted
    """
    length = value.shape[0] #the length of the group
    if(value.iloc[length-1]['SIZE'] > 0):
        value_new = value.iloc[0:length-1] # get rid of the last row if it is valid
    else:
        value_new = value
    cancel_df = value_new.loc[value_new.loc[:,'SIZE']>0,:]
    return cancel_df

def merge_cancel(value):
    """
    For given Timestamp and price, merge the sizes of all the orders
    """
    cancel_merge_df = pd.DataFrame(columns=value.columns)
    cancel_merge_df = cancel_merge_df.append(value.iloc[0])
    cancel_merge_df.iloc[0]['SIZE'] = value['SIZE'].sum()
    return cancel_merge_df

def get_cancel_order(order_filename):
    """
    get cancelled orders from 'order_filename', save these orders in tgt_path
    """
    src_path = os.path.join('RMD/'+order_filename)
    tgt_path = os.path.join('RMD/'+order_filename.replace('Raw','Cancel'))
    trd_path = os.path.join('RMD/'+ order_filename.replace('Order_Raw','TRD'))


    example = pd.read_excel(src_path)
    trade = pd.read_excel(trd_path).reindex(columns=['Time','SIZE','PRICE']).set_index(keys=['Time','PRICE'])

    exm_cancel = example.groupby(['ORDER_ID'],as_index=False)
    cancel_trd = exm_cancel.apply(detect_cancel) \
                .reset_index().drop(columns=['level_0','level_1'])
    cancel_mrg = cancel_trd.groupby(['Time','PRICE'],as_index=False).apply(merge_cancel) \
                .reset_index().drop(columns=['level_0','level_1'])

    cancel_ord = cancel_mrg.join(trade,on=['Time','PRICE'],rsuffix='_trd')\
                .fillna(value={'SIZE_trd':0})
    cancel_ord['SIZE'] = cancel_ord['SIZE'] - cancel_ord['SIZE_trd']
    cancel_ord.drop(columns='SIZE_trd',inplace=True)
    cancel_ord = cancel_ord.loc[cancel_ord['SIZE']>0,:]
    cancel_ord.to_excel(tgt_path)


if __name__ == '__main__':
    get_cancel_order('PN_Order_Raw_080116.xlsx')
