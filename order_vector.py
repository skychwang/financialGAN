import pdb
import sys
import openpyxl
import datetime
import numpy as np

wb = openpyxl.load_workbook(sys.argv[1])
sheet = wb.worksheets[0]

time_interval = int(sys.argv[2]) #in milliseconds
trading_interval = 6.5 * 3600 * 1000
num_intervals = int(trading_interval / time_interval)

# TO-DO preprocessing to find min_max range
buy_min = 650
buy_max = 1150
buy_vector = np.zeros((buy_max - buy_min, num_intervals))
sell_min = 950
sell_max = 1450
sell_vector = np.zeros((sell_max - sell_min, num_intervals))


time_start = datetime.datetime(2016, 8, 10, 9, 30, 0, 000000)
time_end = time_start + datetime.timedelta(microseconds = time_interval * 1000)
i = 2
index = 0
while i < 1000:
	time_stamp = datetime.datetime.strptime(str(sheet["C" + str(i)].value), '%Y/%m/%d %H:%M:%S.%f')
	if time_stamp < time_start:
		i = i + 1
	elif time_start <= time_stamp < time_end:
		if sheet["P" + str(i)].value != 0:
			if sheet["G" + str(i)].value == 0 and buy_min <= sheet["Q" + str(i)].value * 100 < buy_max:
				buy_vector[int(sheet["Q" + str(i)].value * 100 - buy_min), index] = buy_vector[int(sheet["Q" + str(i)].value * 100 - buy_min), index] + \
				sheet["P" + str(i)].value
			elif sheet["G" + str(i)].value == 1 and sell_min <= sheet["Q" + str(i)].value * 100 < sell_max:
				sell_vector[int(sheet["Q" + str(i)].value * 100 - sell_min), index] = sell_vector[int(sheet["Q" + str(i)].value * 100 - sell_min), index] + \
				sheet["P" + str(i)].value
		i = i + 1
	else:
		index = index + 1
		time_start = time_end
		time_end = time_start + datetime.timedelta(microseconds = time_interval * 1000)
pdb.set_trace()