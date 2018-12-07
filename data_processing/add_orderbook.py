from market import *
import numpy as np
import os
import ast


def convert_order(order):
    """
    Convert 6-dim order to 4-dim (type,time,price,quantity) for Class Market
    """
    order_de = np.zeros(4)
    if order[2]<0.5 and order[3]<0.5:
        order_de[0] = 0
    elif order[2]>=0.5 and order[3]<0.5:
        order_de[0] = 1
    elif order[2]<0.5 and order[3]>=0.5:
        order_de[0] = 2
    elif order[2]>=0.5 and order[3]>=0.5:
        order_de[0] = 3

    order_de[1] = order[0]
    order_de[2] = order[4]
    order_de[3] = order[5]

    return order_de

def json2dict(json_dict):
    """
        Convert json_dict ['ASK_PRICE':...] to the format Class Market required [price:quantity]
    """
    dict_buy = {}
    dict_sell = {}
    for i in range(len(json_dict['ASK_PRICE'])):
        dict_sell[json_dict['ASK_PRICE'][i]] = json_dict['ASK_SIZE'][i]

    for i in range(len(json_dict['BID_PRICE'])):
        dict_buy[json_dict['BID_PRICE'][i]] = json_dict['BID_SIZE'][i]
    return [dict_buy, dict_sell]

def get_cda_numpy(folder_name,file_name, initialization=None):
    """
    :param folder_name: folder contains order files
    :param file_name: order file's name
    :param initialization: the initialization state of order book (unusually 9:30) None if unknown
    :return: append 10-level orderbook information
             [order (6 dims)
              orderbook before the order coming (40 dims),
              orderbook after the order coming (40 dims)]
             (86 dims) to order_files by running CDA
    """
    if initialization == None:
        market = Market()
    else:
        order_book_buy = initialization[0]
        order_book_sell = initialization[1]
        market = Market(order_book_sell, order_book_buy)

    orders = np.load(folder_name+"/"+file_name, mmap_mode='r')

    # before_list is the order book state before the current order come
    before_list = []
    # order_list is list of orders
    order_list = []
    # after_list is the order book state after the current order come
    after_list = []

    for i in range(len(orders)):
        # get current order book
        before = market.get_10_level()
        updated = market.update(convert_order(orders[i]))
        # get updated order book
        after = market.get_10_level()

        if np.sum(before) != 0 and np.sum(after) != 0:
            before_list.append(before)
            order_list.append(orders[i])
            after_list.append(after)
            # market.print_market()

    final_file = np.zeros((len(before_list), 86, 1))
    for i in range(len(before_list)):
        before = (before_list[i].reshape((40, 1), order="F"))
        after = (after_list[i].reshape((40, 1), order="F"))
        final = np.concatenate([order_list[i], before, after])
        final_file[i, :, :] = final

    position = [0,1,2,3,4,5,9,10,29,30]
    np.save("cda"+file_name, final_file[:,position,:])

def get_cda_data(folder_name):
    """
        Get process multiple files
    """
    files = os.listdir(folder_name)
    for file in files:
        get_cda_numpy(folder_name, file)

if __name__ == '__main__':
    # get the order book
    with open('PN_OB_080416.json', 'r') as f:
        ob_list_b = ast.literal_eval(f.read())
    # get the order book at 9:30 (initialization)
    current_ob = json2dict(ob_list_b[0])

    get_cda_numpy("RMD", "080416_1.npy", current_ob)
