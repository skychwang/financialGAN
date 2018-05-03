import order_vector_goog


def get_one_day_orders():
    time_interval = 100
    order_filename = "PN_Order_Raw_080116.xlsx"
    order_vector_goog.order_aggregation_one_day(order_filename, time_interval=time_interval)


def get_multiple_days_orders():
    order_vector_goog.order_aggregation_multiple_days()


if __name__ == '__main__':
    get_multiple_days_orders()
