import bisect
import operator
import numpy as np

# This is running the CDA
class Market:
    order_book_sell = {}
    order_book_buy = {}

    lowest_sell = None
    highest_buy = None

    def __init__(self, order_book_sell=None, order_book_buy=None):
        if order_book_sell != None and order_book_buy != None:
            self.order_book_sell = order_book_sell
            self.order_book_buy = order_book_buy
            self.lowest_sell = min(self.order_book_sell)
            self.highest_buy = max(self.order_book_buy)
        else:
            self.order_book_sell = {}
            self.order_book_buy = {}
            self.lowest_sell = None
            self.highest_buy = None

    def get_10_level(self):
        """
            return 10-level orderbook at current time
        """
        order_book = np.zeros((20,2))

        sorted_buy = sorted(self.order_book_buy.items(), key=operator.itemgetter(0))
        sorted_sell = sorted(self.order_book_sell.items(), key=operator.itemgetter(0))

        if len(self.order_book_sell) < 10 or len(self.order_book_buy) < 10:
            print("lack of data")
            return order_book
        else:
            for i in range(10):
                price = sorted_sell[i][0]
                quantity = sorted_sell[i][1]
                order_book[i+10][0] = price
                order_book[i+10][1] = quantity
            for i in range(10):
                price = sorted_buy[len(sorted_buy)-1-i][0]
                quantity = sorted_buy[len(sorted_buy)-1-i][1]
                order_book[i][0] = price
                order_book[i][1] = quantity
            return order_book

    def get_lowest_sell(self):
        """
            return lowest_sell price at current time, if the orderbook is empty, return None
        """
        if self.order_book_sell == {}:
            return None
        return min(self.order_book_sell)

    def get_highest_buy(self):
        """
            return highest_buy price at current time, if the orderbook is empty, return None
        """
        if self.order_book_buy == {}:
            return None
        return max(self.order_book_buy)

    def print_market(self):
        """
            print current market's order book
        """
        print("Current order book")
        print("sell order: (price,quantity) lowest is "+str(self.lowest_sell))
        for order in self.order_book_sell:
            print(str(order) + " " + str(self.order_book_sell[order]))
        print("buy order: (price,quantity) highest is "+str(self.highest_buy))
        for order in self.order_book_buy:
            print(str(order) + " " + str(self.order_book_buy[order]))

    def processing_bid(self, price, quantity):
        """
            process the buy order if it match any sell order in the orderbook
        """
        # if there is no remaining offer, leave the bid in the market
        if self.order_book_sell == {}:
            self.place_bid(self, price, quantity)
            return

        # if the best price in the orderbook has enough quantity the buy order required
        if self.order_book_sell[self.lowest_sell] > quantity:
            self.order_book_sell[self.lowest_sell] -= quantity
            self.lowest_sell = self.get_lowest_sell()
        # if the best price in the orderbook has the same quantity as buy order
        elif self.order_book_sell[self.lowest_sell] == quantity:
            self.order_book_sell.pop(self.lowest_sell)
            self.lowest_sell = self.get_lowest_sell()
        # if the best price in the orderbook do not have enough quantity the buy order required,
        # keep processing with the next best price until there is no available offer
        else:
            remaining_quantity = quantity - self.order_book_sell[self.lowest_sell]
            self.order_book_sell.pop(self.lowest_sell)
            self.lowest_sell = self.get_lowest_sell()

            if self.order_book_sell == {}:
                self.place_bid(price, remaining_quantity)
                return
            if price >= self.lowest_sell:
                self.processing_bid(price, remaining_quantity)
            else:
                self.place_bid(price, remaining_quantity)

    def place_bid(self, price, quantity):
        """
            Put the buy order in it's orderbook there is no matching offer
        """
        if price in self.order_book_buy:
            self.order_book_buy[price] += quantity
        else:
            self.order_book_buy[price] = quantity
        self.highest_buy = self.get_highest_buy()

    def processing_ask(self, price, quantity):
        """
            process the sell order if it match any buy order in the orderbook
        """
        # if there is no remaining offer, leave the ask in the market
        if self.order_book_buy == {}:
            self.place_ask(self, price, quantity)
            return
        # if the best price in the orderbook has enough quantity the sell order required
        if self.order_book_buy[self.highest_buy] > quantity:
            self.order_book_buy[self.highest_buy] -= quantity
            self.highest_buy = self.get_highest_buy()
        # if the best price in the orderbook has the same quantity as sell order
        elif self.order_book_buy[self.highest_buy] == quantity:
            self.order_book_buy.pop(self.highest_buy)
            self.highest_buy = self.get_highest_buy()
        # if the best price in the orderbook do not have enough quantity the sell order required,
        # keep processing with the next best price until there is no available offer
        else:
            remaining_quantity = quantity - self.order_book_buy[self.highest_buy]
            self.order_book_buy.pop(self.highest_buy)
            self.highest_buy = self.get_highest_buy()

            if self.order_book_buy == {}:
                self.place_ask(price, remaining_quantity)
                return
            if price <= self.highest_buy:
                self.processing_ask(price, remaining_quantity)
            else:
                self.place_ask(price, remaining_quantity)

    def place_ask(self, price, quantity):
        """
            Put the sell order in it's orderbook there is no matching offer
        """
        if price in self.order_book_sell:
            self.order_book_sell[price] += quantity
        else:
            self.order_book_sell[price] = quantity
        self.lowest_sell = self.get_lowest_sell()

    def cancel_buy(self, price, quantity):
        """
            Find the possible order for cancel, if not return None
        """
        if self.order_book_buy == {}:
            return None

        # if the price are not available
        if price not in self.order_book_buy:
            # find most similar price in the loop below
            similar_price = list(self.order_book_buy.keys())[0]
            smallest_diff = abs(similar_price - price)
            for order_price in self.order_book_buy:
                diff = abs(order_price-price)
                if diff < smallest_diff:
                    similar_price = order_price
                    smallest_diff = diff
            if smallest_diff <= 0.02:
                price = similar_price
            else:
                return None

        # if there is enough quantity for canceling
        if self.order_book_buy[price] > quantity:
            self.order_book_buy[price] -= quantity
        else:
            quantity = self.order_book_buy[price]
            self.order_book_buy.pop(price)
            self.highest_buy = self.get_highest_buy()

        return price, quantity

    def cancel_sell(self, price, quantity):
        """
            Find the possible order for cancel, if not return None
        """
        if self.order_book_sell == {}:
            return None

        # if the price are not available
        if price not in self.order_book_sell:
            # find most similar price in the loop below
            similar_price = list(self.order_book_sell.keys())[0]
            smallest_diff = abs(similar_price - price)
            for order_price in self.order_book_sell:
                diff = abs(order_price - price)
                if diff < smallest_diff:
                    similar_price = order_price
                    smallest_diff = diff
            if smallest_diff <= 0.02:
                price = similar_price
            else:
                return None


        # if there is enough quantity for canceling
        if self.order_book_sell[price] > quantity:
            self.order_book_sell[price] -= quantity
        else:
            quantity = self.order_book_sell[price]
            self.order_book_sell.pop(price)
            self.lowest_sell = self.get_lowest_sell()

        return price, quantity


    def update(self, order):
        """
            update the current orderbook based on incoming order
            order:[type,time,price,quantity]
        """
        type = order[0]
        price = order[2]
        quantity = order[3]
        # buy order
        if type == 0:
            if self.order_book_sell == {}:
                self.place_bid(price, quantity)
                return
            # if there is a matching offer
            if price >= self.lowest_sell:
                self.processing_bid(price, quantity)
            else:
                self.place_bid(price, quantity)
        # sell order
        elif type == 1:
            if self.order_book_buy == {}:
                self.place_ask(price, quantity)
                return
            # if there is a matching offer
            if price <= self.highest_buy:
                self.processing_ask(price, quantity)
            else:
                self.place_ask(price, quantity)
        # cancel buy order
        elif type == 2:
            return self.cancel_buy(price, quantity)

        # cancel sell order
        elif type == 3:
            return self.cancel_sell(price, quantity)

        return

