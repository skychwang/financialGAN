import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def get_max_min_of_multiple_days():
    """
        Find the min and max price from orderbooks
    """
    # get a list of filenames
    order_books = [file for file in os.listdir("RMD/PN_OB/") if file.startswith("PN_OB")]

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
        prices = get_max_min_of_one_day(order_book)
        print("=========")
        print(prices)

        # update the max and min
        if (current_max_buy < prices[0]):
            current_max_buy = prices[0]
            current_min_buy = prices[1]
            
        if (current_min_sell > prices[3]):
            current_max_sell = prices[2]
            current_min_sell = prices[3]
        max_buy.append(prices[0])
        min_buy.append(prices[1])
        max_sell.append(prices[2])
        min_sell.append(prices[3])


    print("the max of buy is: " + str(current_max_buy))
    print("the min of buy is: " + str(current_min_buy))
    print("the max of sell is: " + str(current_max_sell))
    print("the min of sell is: " + str(current_min_sell))

    # get plot of the change of max-min price over time
    days = np.arange(23)
    plt.plot(days, max_buy, 'go-', label='max_buy')
    plt.plot(days, min_buy, 'yo-', label='min_buy')
    plt.plot(days, max_sell, 'bo-', label='max_sell')
    plt.plot(days, min_sell, 'ro-', label='min_sell')
    plt.legend(loc='upper left')
    plt.xlabel('day')
    plt.ylabel('price')
    plt.show()


def get_max_min_of_one_day(order_filename, price_level=10):
    """
        Return the min and max prices from orderbook in one day
    """
    sheet = pd.read_excel("RMD/PN_OB/" + order_filename)
    level = price_level - 1

    max_buy = np.amax(sheet["BID_PRICE"])
    # get the index of first max
    i = np.amin(np.where(sheet["BID_PRICE"] == max_buy))
    min_buy = sheet["BID_PRICE"][i+level]

    min_sell = np.amin(sheet["ASK_PRICE"])
    # get the index of first min
    i = np.amin(np.where(sheet["ASK_PRICE"] == min_sell))
    max_sell = sheet["BID_PRICE"][i + level]

    return ([max_buy, min_buy, max_sell, min_sell])


if __name__ == '__main__':
    get_max_min_of_multiple_days()