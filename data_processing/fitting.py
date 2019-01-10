from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pylab as plb
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# This file computes distance between distributions

def fit_geo_dist(p, size):
    """
        Given p and size, fit a geometric distribution
    """
    x = np.arange(1, size+1)
    rv = scipy.stats.geom(p)
    fit_pmf = rv.pmf(x)
    return fit_pmf

def plot_geo_dist(time_hist,p,size):
    """
        Plot the geometric distribution pmf
    """
    x = np.arange(1, size + 1)
    plt.plot(x, [time_hist[i - 1] / sum(time_hist) for i in range(1, len(time_hist) + 1)])
    plt.plot(x, scipy.stats.geom.pmf(x, p), 'bo', ms=3, label='geom pmf')
    plt.legend(loc='best', frameon=False)
    plt.show()

def get_geomatric_parameter(time_hist):
    """
        Get maximum likelihood estimates of p of geometric distribution from data
    """
    sum_x = 0
    for i in range(len(time_hist)):
        # i is the number of failures, i+1 is the number of trials
        x = i+1
        # frequency of x number of trials occurs
        frequency = time_hist[i]
        sum_x += x*frequency

    #sum(time_dist) is the number of success, sum_x is the the number of trial needed
    p = sum(time_hist)/sum_x
    return p

def read_real_interval(file):
    """
        read the orders' intervals from real data
    """
    orders = np.load(file, mmap_mode='r')
    interval_list = []
    for order in orders:
        #interval_list.append(int(order[1][0]))
        interval_list.append(int(order[1]))
    return interval_list

def read_fake_interval(file):
    """
        Read the orders' intervals from fake data
    """
    orders = np.load(file, mmap_mode='r')
    interval_list = []
    for order in orders:
        if sum(order) == 0:
            break
        interval_list.append(int(order[1]))

    return interval_list

def plot_normal(price_list, mu, sigma, min=916, max=942):
    """
        Plot the normal distribution
    """
    # Create the bins and histogram
    count, bins, ignored = plt.hist(price_list, bins=np.arange(min, max+0.01, 0.01), normed=True)

    # Plot the distribution curve
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.show()

# read the orders' prices from real data
def read_real_price(file):
    """
        Read the orders' prices from real data
    """
    orders = np.load(file, mmap_mode='r')
    price_list = []

    for order in orders:
        #price_list.append((order[4][0]))
        if sum(order) == 0:
            break
        price_list.append((order[4]))
    return price_list

def read_fake_price(file):
    """
        Read the orders' prices from fake data
    """
    orders = np.load(file, mmap_mode='r')
    price_list = []
    for order in orders:
        if sum(order) == 0:
            break
        price_list.append((order[4]))
    return price_list

def get_type(order):
    """
        Convert the two continuous order types to [0,1,2,3]
    """
    #buy
    if order[2]<0.5 and order[3]<0.5:
        type = 0
    #sell
    elif order[2]>=0.5 and order[3]<0.5:
        type = 1
    #cancel buy
    elif order[2]<0.5 and order[3]>=0.5:
        type = 2
    #cancel sell
    elif order[2]>=0.5 and order[3]>=0.5:
        type = 3
    return type


# real_file is real data
# fake_file is generated data
# min_p and max_p are the actual min and max price from real data
# bin_size is size of bin for price
def test_price(real_file,fake_file, min_p, max_p, bin_size=0.01):
    """
        Compare generated data with fitted data
    """
    # get real distribution
    price_list = read_real_price(real_file)
    hist, bin_edges = np.histogram(price_list, bins=np.arange(min_p, max_p+bin_size, bin_size))
    sum_hist = sum(hist)
	# pdf - not a cdf (forgot to change name)
    real_cdf_dist = [i /sum_hist for i in hist]


    #fit the normal dist
    variance = np.var(price_list)
    sigma = math.sqrt(variance)
    mu = np.mean(price_list)
    price_list = np.random.normal(loc=mu, scale=sigma,size=80000)
    hist, bin_edges = np.histogram(price_list, bins=np.arange(min_p, max_p + bin_size, bin_size))
    sum_hist = sum(hist)
    fitted_dist = ([i / sum_hist for i in hist])

    #plot_normal(price_list,mu, sigma,min=min_p, max=max_p)
    #get difference between real and fitted
    print("Price: real-fit")
    print(sum([abs(real_cdf_dist[i] - fitted_dist[i]) for i in range(len(fitted_dist))]))

    # get fake distribution
    price_list = read_fake_price(fake_file)
    hist, bin_edges = np.histogram(price_list, bins=np.arange(min_p, max_p+bin_size, bin_size))
    sum_hist = sum(hist)
    fake_cdf_dist = ([i /sum_hist for i in hist])

    #get difference between real and generated
    print("Price: real-fake")
    print(sum([abs(fake_cdf_dist[i]-real_cdf_dist[i]) for i in range(len(fitted_dist))]))


# interarrival time
def test_interval(real_file,fake_file,max_interval=1000):
    """
        Compare generated data with fitted data
    """
    # get real distribution
    interval_list = read_real_interval(real_file)
    hist, bin_edges = np.histogram(interval_list, bins=np.arange(0, max(max(interval_list),max_interval), 1))
    sum_hist = sum(hist)
    normalized_hist = [i /sum_hist for i in hist]
    real_cdf_dist = (normalized_hist)[:max_interval]

    #get fitted geo dist
    p = get_geomatric_parameter(hist[:max_interval])
    fitted_dist = (fit_geo_dist(p, max_interval))

    #plot_geo_dist(hist[:1000], p, 1000)
	# distance between fitted and real
    print("Interval: real-fit")
    print(sum([abs(fitted_dist[i]-real_cdf_dist[i]) for i in range(len(fitted_dist))]))

    # get fake distribution
    interval_list = read_fake_interval(fake_file)
    hist, bin_edges = np.histogram(interval_list, bins=np.arange(0, max(max(interval_list),max_interval), 1))
    sum_hist = sum(hist)
    normalized_hist = [i / sum_hist for i in hist]
    fake_cdf_dist = (normalized_hist)[:max_interval]


	# distance between generated and real
    print("Interval: real-fake")
    print(sum([abs(fake_cdf_dist[i]-real_cdf_dist[i]) for i in range(len(fitted_dist))]))


#test_price("real_ob/080117_1_adjusted.npy","predict_goog21_new0_5000_1_full_day.npy",916,942)
#test_interval("real_ob/080117_1_adjusted.npy","predict_goog21_new0_5000_1_full_day.npy",1000)


#test_price("real_ob/data_pn.npy","real_ob/predict_PN_generated.npy",6,13)
#test_interval("real_ob/data_pn.npy","real_ob/predict_PN.npy",100)

#test_price("real_data_syn.npy","syn_gen.npy",-1,1)
#test_interval("real_data_syn.npy","syn_gen.npy",30)

#test_price("real_ob/KS/syn_real_buy.npy","real_ob/KS/syn_fake_buy.npy",-1,1)
#test_interval("real_ob/KS/syn_real_buy.npy","real_ob/KS/syn_real_buy.npy",30)


#test_price("real_ob/KS/goog_real_sell (1).npy","real_ob/KS/goog_fake_sell (1).npy",916,942)
#test_price("real_ob/KS/goog_real_buy(1).npy","real_ob/KS/goog_fake_buy(1).npy",916,942)



"""

test_price("real_ob/KS/goog_real_buy.npy","real_ob/KS/goog_fake_buy.npy",916,942,0.005)
test_price("real_ob/KS/goog_real_sell.npy","real_ob/KS/goog_fake_sell.npy",916,942,0.005)


test_price("real_ob/KS/pn_real_buy.npy","real_ob/KS/pn_fake_buy.npy",6,13,0.005)
test_price("real_ob/KS/pn_real_sell.npy","real_ob/KS/pn_fake_sell.npy",6,13,0.005)


"""



test_price("real_ob/KS/syn_real_buy.npy","real_ob/KS/syn_fake_buy.npy",-1,1,0.0001)

test_price("real_ob/KS/syn_real_sell.npy","real_ob/KS/syn_fake_sell.npy",-1,1,0.0001)

#test_interval("real_ob/KS/pn_real_buy.npy","real_ob/KS/pn_fake_buy.npy",100)
#test_interval("real_ob/KS/pn_real_sell.npy","real_ob/KS/pn_fake_sell.npy",100)



#test_interval("real_ob/KS/goog_real_buy.npy","real_ob/KS/goog_fake_buy.npy",1000)
#test_interval("real_ob/KS/goog_real_sell.npy","real_ob/KS/goog_fake_sell.npy",1000)


#test_interval("real_ob/KS/syn_real_buy.npy","real_ob/KS/syn_fake_buy.npy",30)
#test_interval("real_ob/KS/syn_real_sell.npy","real_ob/KS/syn_fake_sell.npy",30)
