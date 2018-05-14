import matplotlib.pyplot as plt
from read_json import *
from order_vector import *
import operator


def get_size_distribution_per_price_level(orderstreams, is_buy=True, mode="average",
                                          sell_min=900, sell_max=1500,
                                          buy_min=600, buy_max=1200):
    #print(orderstreams.shape)
    num_of_batch = orderstreams.shape[0]

    num_batches = orderstreams.shape[0]
    batch_size = orderstreams.shape[1]
    order_stream_size = orderstreams.shape[2]
    if (is_buy):
        price_range = buy_max - buy_min
    else:
        price_range = sell_max - sell_min

    # convert the orderstream to a array of mutiple orders
    orderstreams = np.reshape(orderstreams, (num_batches*batch_size*order_stream_size, price_range, 2))

    buy_vector = orderstreams[:, :, 0]
    sell_vector = orderstreams[:, :, 1]
    #print(buy_vector.shape)

    # get x-axis
    if (is_buy):
        x = np.arange(buy_min, buy_max, 1.0)
        input_vector = buy_vector
    else:
        x = np.arange(sell_min, sell_max, 1.0)
        input_vector = sell_vector

    # get y-axis based on mode
    if (mode=="average"):
        #get the average size of each price
        y = np.mean(input_vector, axis=0)
    elif (mode=="median"):
        # get the median size of each price
        y = np.median(input_vector, axis=0)
    elif (mode=="sum"):
        # get the summation size of each price
        y = np.sum(input_vector, axis=0)
    elif (mode=="max"):
        #get the max size of each price
        y = np.amax(input_vector, axis=0)
    elif (mode=="min"):
        # get the min size of each price
        y = np.amin(input_vector, axis=0)
    elif (mode=="non_zero"):
        # get the min size of each price
        y = np.count_nonzero(input_vector, axis=0)
    elif (mode=="all"):
        # get the all size of each price
        fig = plt.figure()
        num_orders = input_vector.shape[0]

        x = []
        y = []
        for i in range(num_orders):
            if (is_buy):
                x += np.arange(buy_min, buy_max, 1).tolist()
            else:
                x += np.arange(sell_min, sell_max, 1).tolist()

            y += input_vector[i].tolist()

        plt.scatter(x, y, marker=".", s=1)

        if (is_buy):
            plt.xlabel("price(buy vector)")
        else:
            plt.xlabel("price(sell vector)")
        plt.ylabel(mode + " order size at this price")
        plt.legend(loc=1)
        if (is_buy):
            fig.savefig('(buy)all_size_distribution_per_price.png', dpi=200)
        else:
            fig.savefig('(sell)all_size_distribution_per_price.png', dpi=200)


        return

    # print(y)

    # print(np.amax(sell_vector))

    fig = plt.figure()


    plt.scatter(x, y, marker=".", s=1)
    if (is_buy):
        plt.xlabel("price(buy vector)")
    else:
        plt.xlabel("price(sell vector)")

    plt.ylabel(mode+" order size at this price")
    plt.legend(loc=1)
    plt.title(mode+' order size at each price level')
    if (is_buy):
        fig.savefig("(buy)"+mode + 'order_size_at_each_price_level.png', dpi=200)
    else:
        fig.savefig("(sell)"+mode + 'order_size_at_each_price_level.png', dpi=200)


def get_nonzero_element_distribution(orderstreams, is_buy=True, mode="nonzero"):
    num_of_batch = orderstreams.shape[0]
    num_batches = orderstreams.shape[0]
    batch_size = orderstreams.shape[1]
    order_stream_size = orderstreams.shape[2]

    orderstreams = np.reshape(orderstreams, (num_batches * batch_size * order_stream_size * 600, 2))
    buy_vector = orderstreams[:, 0]
    sell_vector = orderstreams[:, 1]

    if (is_buy):
        input_vector = buy_vector
    else:
        input_vector = sell_vector

    if (mode!="nonzero"):
        hist, bin_edges = np.histogram(input_vector, bins=int(np.amax(input_vector)))
        # print(hist)
        # print(bin_edges)
        num = len(input_vector)
        print("#######################################")
        print("the distribution of nonzero element are")
        for i in range(len(hist)):
            if (hist[i] != 0):
                print("size " + str(bin_edges[i]) + " show " + str(hist[i]) + " times" + \
                      ", which is " + str(hist[i] / num * 100) + "% of all")
    else:
        hist, bin_edges = np.histogram(input_vector, bins=np.arange(1,np.amax(input_vector),1))
        fig = plt.figure()
        plt.hist(input_vector, bins=np.arange(1,2000,1))
        plt.xlabel("size")
        plt.ylabel("frequency")
        plt.title("the hist of size freqency")
        if (is_buy):
            fig.savefig('(buy)the_hist_of_size_freqency.png', dpi=400)
        else:
            fig.savefig('(sell)the_hist_of_size_freqency.png.png', dpi=400)
        # print(hist)
        # print(bin_edges)
        num_nonzero = np.count_nonzero(input_vector)
        print("#######################################")
        print("the distribution of nonzero element are")
        for i in range(len(hist)):
            if (hist[i] != 0):
                print("size " + str(bin_edges[i]) + " show " + str(hist[i]) + " times" + \
                      ", which is " + str(hist[i] / num_nonzero * 100) + "% of all")




def get_nonzero_vector_distribution(orderstreams, is_buy=True, mode="hist",
                                    sell_min=900, sell_max=1500,
                                    buy_min=600, buy_max=1200):
    #print(orderstreams.shape)
    num_of_batch = orderstreams.shape[0]

    num_batches = orderstreams.shape[0]
    batch_size = orderstreams.shape[1]
    order_stream_size = orderstreams.shape[2]
    if (is_buy):
        price_range = buy_max - buy_min
    else:
        price_range = sell_max - sell_min

    # convert the orderstream to a array of mutiple orders
    orderstreams = np.reshape(orderstreams, (num_batches * batch_size * order_stream_size, price_range, 2))

    buy_vector = orderstreams[:, :, 0]
    sell_vector = orderstreams[:, :, 1]
    #print(buy_vector.shape)

    # get x-axis
    x = np.arange(0, orderstreams.shape[0], 1.0)
    if (is_buy):
        input_vector = buy_vector
    else:
        input_vector = sell_vector

    # get the number of nonzero element at each time interval(vector)
    y = np.count_nonzero(input_vector, axis=1)
    num_vectors = input_vector.shape[0]
    fig = plt.figure()
    if (mode == "hist"):
        plt.hist(y, bins=int(np.amax(y)))
        hist, bin_edges = np.histogram(y, bins=int(np.amax(y)))
        #print(hist)
        #print(bin_edges)
        print("#############################################################")
        print("the distribution of nonzero element in each time interval are")
        for i in range(len(hist)):
            if (hist[i] != 0):
                print(str(hist[i]) + " vectors has " + str(bin_edges[i]) + " nonzero element" + \
                      ", which is " + str(hist[i] / num_vectors*100) + "%")


    elif (mode=="hist_nonzero"):
        # get rid of all zeros
        #input_vector = input_vector[np.nonzero(input_vector)]
        #print(input_vector)
        #print(input_vector.shape)
        y = np.count_nonzero(input_vector, axis=1)

        plt.hist(y, bins=np.arange(1, np.amax(y),1))
        hist, bin_edges = np.histogram(y, bins=np.arange(1, np.amax(y),1))
        #print(hist)
        #print(bin_edges)
        print("#############################################################")
        print("the distribution of nonzero element in each time interval are")
        for i in range(len(hist)):
            if (hist[i]!=0):
                print(str(hist[i]) + " vectors has " + str(bin_edges[i]) + " nonzero element" + \
                      ", which is " + str(hist[i] / num_vectors*100) + "%")

    elif (mode == "scatter"):
        plt.plot(x, y, ".")
        plt.xlabel("order vectors(time intervel of 100ns)")

    elif (mode=="all"):
        # get the all size of each price
        fig = plt.figure()

        x = []
        y = []
        for i in range(price_range):
            x += np.arange(0, orderstreams.shape[0],1).tolist()
            temp = np.asarray(input_vector[:, i].tolist())
            temp[temp!=0] = i
            y += (temp.tolist())

        plt.scatter(x, y, marker=".", s=1)

        plt.xlabel("100ns time(each vector)")
        plt.ylabel("price distribution")
        plt.legend(loc=1)
        if (is_buy):
            fig.savefig('(buy)all_price_distribution_per_order.png', dpi=200)
        else:
            fig.savefig('(sell)all_price_distribution_per_order.png', dpi=200)


        return

    #print(y)
    plt.ylabel("number of nonzero element")
    plt.legend(loc=1)
    plt.title(mode + ' of nonzero distribution')
    if (is_buy):
        fig.savefig("(buy)"+mode + ' of nonzero distribution.png', dpi=200)
    else:
        fig.savefig("(sell)"+mode + ' of nonzero distribution.png', dpi=200)


# get the distribution of interval of two nonzero element
def get_time_interval_distribution(orderstreams, is_buy=True, mode="hist",
                                    sell_min=900, sell_max=1500,
                                    buy_min=600, buy_max=1200):
    #print(orderstreams.shape)
    num_of_batch = orderstreams.shape[0]

    num_batches = orderstreams.shape[0]
    batch_size = orderstreams.shape[1]
    order_stream_size = orderstreams.shape[2]
    if (is_buy):
        price_range = buy_max - buy_min
    else:
        price_range = sell_max - sell_min

    # convert the orderstream to a array of mutiple orders
    orderstreams = np.reshape(orderstreams, (num_batches * batch_size * order_stream_size, price_range, 2))

    buy_vector = orderstreams[:, :, 0]
    sell_vector = orderstreams[:, :, 1]
    #print(buy_vector.shape)

    # get x-axis
    x = np.arange(0, orderstreams.shape[0], 1.0)
    if (is_buy):
        input_vector = buy_vector
    else:
        input_vector = sell_vector

    # get a summation of the size varies with time
    y = np.sum(buy_vector, axis=1)

    #print(y)
    fig = plt.figure()

    if (mode=="hist"):
        last_nonzero = 0
        list = []
        # get the time interval of all nonzero element
        for i in range(1, y.shape[0]):
            if (y[i] != 0):
                list.append(i - last_nonzero - 1)
                last_nonzero = i

        #print(list)
        plt.hist(list, bins=np.arange(0, max(list), 1))

        hist, bin_edges = np.histogram(list, bins=np.arange(0, max(list), 1))
        plt.xlabel("number of 0 before nonzero vector appear")
        plt.ylabel("frequency")
        plt.legend(loc=1)
        plt.title('number of 0 before nonzero vector distribution')
        if (is_buy):
            fig.savefig('(buy)time_interval_distribution.png', dpi=200)
        else:
            fig.savefig('(sell)time_interval_distribution.png', dpi=200)
        print("###################################################################")
        print("the distribution of the time interval between nonzero vector are")

        for i in range(len(hist)):
            if (hist[i] != 0):
                print(str(hist[i]) + " nonzero vectors has " + str(bin_edges[i]) + " all zero vectors before it" + \
                      ", which is " + str(hist[i] / len(list)*100) + "%")
    else:
        #print(x.shape)
        #print(y[np.nonzero(y)].shape)
        plt.scatter(x, (y > 0), s=1)
        plt.show()




def get_distribution(orderstreams):
    # buy
    print("The distribution for buy vectors are")

    get_size_distribution_per_price_level(orderstreams, mode="max")

    get_size_distribution_per_price_level(orderstreams, mode="average")
    get_size_distribution_per_price_level(orderstreams, mode="non_zero")
    #get_size_distribution_per_price_level(orderstreams, mode="all")

    get_nonzero_element_distribution(orderstreams, mode="inclue_zero")
    get_nonzero_element_distribution(orderstreams)

    get_nonzero_vector_distribution(orderstreams, mode="hist")
    get_nonzero_vector_distribution(orderstreams, mode="hist_nonzero")
    # get_nonzero_vector_distribution(orderstreams, mode="all")

    get_time_interval_distribution(orderstreams, mode="hist")




    #sell
    print("The distribution for sell vectors are")
    get_size_distribution_per_price_level(orderstreams,is_buy=False, mode="max")

    get_size_distribution_per_price_level(orderstreams,is_buy=False, mode="average")
    get_size_distribution_per_price_level(orderstreams,is_buy=False, mode="non_zero")
    # get_size_distribution_per_price_level(orderstreams,is_buy=False, mode="all")

    get_nonzero_element_distribution(orderstreams,is_buy=False, mode="inclue_zero")
    get_nonzero_element_distribution(orderstreams,is_buy=False)

    get_nonzero_vector_distribution(orderstreams,is_buy=False, mode="hist")
    get_nonzero_vector_distribution(orderstreams,is_buy=False, mode="hist_nonzero")
    # get_nonzero_vector_distribution(orderstreams,is_buy=False, mode="all")

    get_time_interval_distribution(orderstreams,is_buy=False, mode="hist")


#orderstreams should be (num_time_intervals, 2)
def test_zero_one_distribution(orderstreams, is_buy=True):
    #print(orderstreams.shape)
    total_time_intervals = orderstreams.shape[0]
    buy_vector = orderstreams[:, 0]
    sell_vector = orderstreams[:, 1]
    #print(buy_vector.shape)

    if (is_buy):
        input_vector = buy_vector
    else:
        input_vector = sell_vector

    print("the percentage of nonzero are")
    print(str((np.count_nonzero(input_vector) / total_time_intervals) * 100) + "%")

    y = input_vector

    last_nonzero = 0
    list = []
    # get the time interval of all nonzero element
    for i in range(1, total_time_intervals):
        if (y[i] != 0):
            list.append(i - last_nonzero - 1)
            last_nonzero = i

    # print(list)
    fig = plt.figure()

    plt.hist(list, bins=np.arange(0, max(list), 1))

    hist, bin_edges = np.histogram(list, bins=np.arange(0, max(list), 1))
    plt.xlabel("number of 0 before nonzero vector appear")
    plt.ylabel("frequency")
    plt.legend(loc=1)
    plt.title('number of 0 before nonzero vector distribution')
    if (is_buy):
        fig.savefig('fake_buy_0-1.png', dpi=200)
    else:
        fig.savefig('fake_sell_0-1.png', dpi=200)

    print("###################################################################")
    print("the distribution of the time interval between nonzero vector are")

    for i in range(len(hist)):
        if (hist[i] != 0):
            print(str(hist[i]) + " nonzero vectors has " + str(bin_edges[i]) + " all zero vectors before it" + \
                  ", which is " + str(hist[i] / len(list) * 100) + "%")



if __name__ == '__main__':
    orderstreams = np.load("predict_13.npy", mmap_mode='r')
    #get_distribution(orderstreams)

    print(orderstreams.shape)

    total_time_intervals = orderstreams.shape[0] * orderstreams.shape[1]
    orderstreams = np.reshape(orderstreams,(total_time_intervals,1))
    orderstreams = orderstreams[600:]

    print(orderstreams)

    orderstream = np.zeros((total_time_intervals, 1))

    for i in range(orderstreams.shape[0]):
        if (orderstreams[i]<=0.5):
            orderstream[i] = 0
        else:
            orderstream[i] = 1

    orderstream = orderstream.astype(int)
    #np.set_printoptions(threshold=np.nan)
    """
    np.set_printoptions(threshold=np.nan)
    print(orderstreams.shape)

    num_batches = orderstreams.shape[0]
    batch_size = orderstreams.shape[1]
    order_stream_size = orderstreams.shape[2]

    total_time_intervals = num_batches * batch_size * order_stream_size

    orderstreams = np.reshape(orderstreams, (total_time_intervals, 2))

    """
    orderstream_copy = np.reshape(orderstream, (total_time_intervals, 1))
    orderstream = np.concatenate((orderstream_copy, orderstream_copy), axis=1)
    print(orderstream.shape)
    test_zero_one_distribution(orderstream)

    #get_average_size_distribution_per_price_level(orderstreams, mode="max")
    #get_nonzero_vector_distribution(orderstreams, mode="scatter")
    #get_time_interval_distribution(orderstreams, mode="hist")






