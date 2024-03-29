import pandas as pd
import numpy as np
import os

# extract cancellations from real data
def detect_cancel(value):
    """
    Find orders that are cancelled or transacted
    """
    length = value.shape[0] #the length of the group
    cancel_df = pd.DataFrame(columns=value.columns)
    for i in range(1,length):
		# if the quantity for this order id decreased, then this is a candidate cancellation
        if(value.iloc[i]['SIZE']<value.iloc[i-1]['SIZE']):
            tmp_order = value.iloc[i]
            tmp_order['SIZE'] = value.iloc[i-1]['SIZE'] - value.iloc[i]['SIZE']
            cancel_df = cancel_df.append(tmp_order)
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
    #path of order file
    src_path = os.path.join('GOOG/'+order_filename)
    #path to save the extracted cancel orders
    tgt_path = os.path.join('GOOG/'+order_filename.replace('Raw','Cancel'))
    #path of transaction file
    trd_path = os.path.join('GOOG/'+ order_filename.replace('Order_Raw','TRD'))

    #read orders
    example = pd.read_excel(src_path)
    #read transactions
    trade = pd.read_excel(trd_path).reindex(columns=['Time','SIZE','PRICE'])\
        .set_index(keys=['Time','PRICE'])

    #Group orders by order_id
    exm_cancel = example.groupby(['ORDER_ID'],as_index=False)
    #Find orders transacted or cancelled
    cancel_trd = exm_cancel.apply(detect_cancel) \
                .reset_index().drop(columns=['level_0','level_1'])
    #Merge All orders at the same price and time
    cancel_mrg = cancel_trd.groupby(['Time','PRICE'],as_index=False).apply(merge_cancel) \
                .reset_index().drop(columns=['level_0','level_1'])
    #Get transacted orders by time and price
    cancel_ord = cancel_mrg.join(trade,on=['Time','PRICE'],rsuffix='_trd')\
                .fillna(value={'SIZE_trd':0})
    #Subtract transaction
    cancel_ord['SIZE'] = cancel_ord['SIZE'] - cancel_ord['SIZE_trd']
    cancel_ord.drop(columns='SIZE_trd',inplace=True)
    #Keep cancel order size greater than 0
    cancel_ord = cancel_ord.loc[cancel_ord['SIZE']>0,:]
    cancel_ord.to_excel(tgt_path,index=False)

def order_cancel_multiple_days():
    """
    Function to get cancel orders for multiple days
    """
	raw_orders = [file for file in os.listdir("GOOG/") if file.startswith("GOOG_Order")]
	for raw_order in raw_orders:
		get_cancel_order(raw_order)

if __name__ == '__main__':
    order_cancel_multiple_days()
