# 10-level real order book used in training
import os
import pandas as pd

#[order_book_buy, order_book_sell, highest_buy, lowest_sell]
def get_market_list(bid_price, bid_size, ask_price, ask_size):
    order_book_buy = {}
    order_book_sell = {}

    for i in range(len(bid_price)):
        order_book_buy[bid_price[i]] = bid_size[i]
    highest_buy = max(order_book_buy)

    for i in range(len(ask_price)):
        order_book_sell[ask_price[i]] = ask_size[i]
    lowest_sell = min(order_book_sell)

    return [order_book_buy, order_book_sell, highest_buy, lowest_sell]

# 11:49:16.230 -> 114916230
def time_str2num(time_str):
    num = 0
    time_list = time_str.split(":")
    print(time_list)
    num += 10000000*int(time_list[0])
    num += 100000*int(time_list[1])
    num += 1000*int(time_list[2].split(".")[0])
    num += int(time_list[2].split(".")[1])

    return num

# given date, hour, min,
class Real_market:
    order_book_sell = {}
    order_book_buy = {}

    lowest_sell = None
    highest_buy = None

    def __init__(self, bid_price, bid_size, ask_price, ask_size):
        for i in range(len(bid_price)):
            self.order_book_buy[bid_price[i]] = bid_size[i]

        self.highest_buy = max(self.order_book_buy)

        for i in range(len(ask_price)):
            self.order_book_sell[ask_price[i]] = ask_size[i]
        self.lowest_sell = min(self.order_book_sell)


class Real_order_book:
    # {date:{time: Real_market}}
    order_book = {}

    def __init__(self, file_path, num_price_level=10):
        for file in os.listdir(file_path):
            sheet = pd.read_excel(file_path + file)

            num_orderbooks = int(len(sheet)/(num_price_level+1))
            print(num_orderbooks)

            for i in range(num_orderbooks):
                index = i * 11
                time_list = sheet["Time"][index].split(" ")
                date = time_list[0]
                time = time_list[1]
                bid_price = sheet["BID_PRICE"][index:index+num_price_level].tolist()
                bid_size = sheet["BID_SIZE"][index:index+num_price_level].tolist()
                ask_price = sheet["ASK_PRICE"][index:index+num_price_level].tolist()
                ask_size = sheet["ASK_SIZE"][index:index+num_price_level].tolist()
                time = time_str2num(time)
                if date not in self.order_book:
                    self.order_book[date] = {}
                self.order_book[date][time] = get_market_list(bid_price, bid_size,
                                                              ask_price, ask_size)


    def find_order_book(self, date, time):
        time = time_str2num(time)
        closest_time = time
        if time not in self.order_book[date]:
            diff = time
            for avaliable_time in self.order_book[date]:
                if avaliable_time <= time and abs(time - avaliable_time) < diff:
                    print("avaliable_time")
                    print(avaliable_time)
                    diff = abs(time - avaliable_time)
                    closest_time = avaliable_time
        return self.order_book[date][closest_time]


if __name__ == '__main__':
    order_book = Real_order_book("RMD/PN_OB/")
    # [order_book_buy, order_book_sell, highest_buy, lowest_sell]
    print(order_book.find_order_book("2016/08/05", "09:30:00.034"))






