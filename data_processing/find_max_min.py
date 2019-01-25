import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_max_min_of_multiple_days(path):
    """
        Find the min and max price from orderbooks
    """
    # get a list of filenames
    order_books = [file for file in os.listdir(path) if file.startswith("PN_OB")]

    # set initial value of max and min
    current_max_buy = 0
    current_min_buy = np.inf
    current_max_sell = 0
    current_min_sell = np.inf
    max_buy = []
    min_buy = []
    max_sell = []
    min_sell = []

    for order_book in order_books:
        # get [max_buy, min_buy, max_sell, min_sell] from one day
        prices = get_max_min_of_one_day(path+order_book)
        print("=========")
        print(prices)

        # update the max and min
        if (current_max_buy < prices[0]):
            current_max_buy = prices[0]

        if (current_min_buy > prices[1]):
            current_min_buy = prices[1]
            
        if (current_min_sell > prices[3]):
            current_min_sell = prices[3]

        if (current_max_sell < prices[2]):
            current_max_sell = prices[2]

        max_buy.append(prices[0])
        min_buy.append(prices[1])
        max_sell.append(prices[2])
        min_sell.append(prices[3])

    print("the max of buy is: " + str(current_max_buy))
    print("the min of buy is: " + str(current_min_buy))
    print("the max of sell is: " + str(current_max_sell))
    print("the min of sell is: " + str(current_min_sell))

def get_max_min_of_one_day(order_filename):
    """
        Return the min and max prices from orderbook in one day
    """
    sheet = pd.read_excel(order_filename)
    max_buy = np.amax(sheet["BID_PRICE"])
    min_buy = np.amin(sheet["BID_PRICE"])
    min_sell = np.amin(sheet["ASK_PRICE"])
    max_sell = np.amax(sheet["ASK_PRICE"])

    return ([max_buy, min_buy, max_sell, min_sell])


if __name__ == '__main__':
    get_max_min_of_multiple_days("RMD/PN_OB/")