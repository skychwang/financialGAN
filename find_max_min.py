import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def get_max_min_of_multiple_days():
    order_books = [file for file in os.listdir("RMD/") if file.startswith("PN_OB")]
    current_max_buy = 0
    current_min_buy = np.inf
    current_max_sell = 0
    current_min_sell = np.inf
    max_buy = []
    min_buy = []
    max_sell = []
    min_sell = []
    for order_book in order_books:
        print("=========")
        temp = get_max_min_of_one_day(order_book)
        current_max_buy = max(current_max_buy, temp[0])
        current_min_buy = min(current_min_buy, temp[1])
        current_max_sell = max(current_max_sell, temp[2])
        current_min_sell = min(current_min_sell, temp[3])
        max_buy.append(temp[0])
        min_buy.append(temp[1])
        max_sell.append(temp[2])
        min_sell.append(temp[3])
    print("the max of buy is: " + str(current_max_buy))
    print("the min of buy is: " + str(current_min_buy))
    print("the max of sell is: " + str(current_max_sell))
    print("the min of sell is: " + str(current_min_sell))

    days = np.arrange(23)
    plt.plot(days, max_buy, 'go-', label='max_buy')
    plt.plot(days, min_buy, 'yo-', label='min_buy')
    plt.plot(days, max_sell, 'bo-', label='max_sell')
    plt.plot(days, min_sell, 'ro-', label='min_sell')
    plt.legend(loc='upper left')
    plt.xlabel('day')
    plt.ylabel('price')
    plt.show()


def get_max_min_of_one_day(order_filename):
    sheet = pd.read_excel("RMD/" + order_filename)
    max_buy = np.amax(sheet["BID_PRICE"])
    min_buy = np.amin(sheet["BID_PRICE"])
    max_sell = np.amax(sheet["ASK_PRICE"])
    min_sell = np.amin(sheet["ASK_PRICE"])
    print([max_buy, min_buy, max_sell, min_sell])
    return ([max_buy, min_buy, max_sell, min_sell])


if __name__ == '__main__':
    get_max_min_of_multiple_days()
