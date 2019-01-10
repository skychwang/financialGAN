import pdb
import sys
import openpyxl
import datetime
import numpy as np
import os
import pandas as pd

# This file converts from excel to json
def order_aggregation_multiple_days():
	# filename will change with different stock
	raw_orders = [file for file in os.listdir("RMD/") if file.startswith("PN_Order_Raw")]
	for raw_order in raw_orders:
		order_aggregation_one_day("output/",raw_order, time_interval=1)
	raw_orders = [file for file in os.listdir("RMD/") if file.startswith("PN_Order_Cancel")]
	for raw_order in raw_orders:
		order_aggregation_one_day("output_cancel/",raw_order, time_interval=1)


def order_aggregation_one_day(out_dir,order_filename, time_interval=100):
	sheet = pd.read_excel("RMD/" + order_filename)
	time_vector = []

	# TO-DO preprocessing to find min_max range 600 1200 900 1500
	# This should come from find_max_min.py
	buy_min = 600
	buy_max = 1200
	buy_vector = []
	sell_min = 900
	sell_max = 1500
	sell_vector = []

	# month date and year from the filename
	parse_time = order_filename.split("_")[3].split(".")[0]
	month = int(parse_time[:2])
	date = int(parse_time[2:4])
	year = 2000 + int(parse_time[4:6])
	save_filename =  out_dir + parse_time + "_" + str(time_interval) + ".json"

	
	time_start = datetime.datetime(year, month, date, 9, 30, 0, 000000)
	time_end = time_start + datetime.timedelta(microseconds = time_interval * 1000)

	i = 0
	buy_dict = {}
	sell_dict = {}
	#len(sheet)
	while i < len(sheet):
		if i%100 == 0:
			print("processed {} lines".format(i))
			print(datetime.datetime.strptime(sheet["Time"][i], '%Y/%m/%d %H:%M:%S.%f'))
			print(time_start)
			print(time_end)
		time_stamp = datetime.datetime.strptime(sheet["Time"][i], '%Y/%m/%d %H:%M:%S.%f')
		if time_stamp < time_start:
			i = i + 1
		elif time_start <= time_stamp < time_end:
			if sheet["SIZE"][i] != 0:
				if sheet["BUY_SELL_FLAG"][i] == 0 and buy_min <= sheet["PRICE"][i] * 100 < buy_max:
					if int(sheet["PRICE"][i] * 100 - buy_min) in buy_dict:
						buy_dict[int(sheet["PRICE"][i] * 100 - buy_min)] += sheet["SIZE"][i]
					else:
						buy_dict[int(sheet["PRICE"][i] * 100 - buy_min)] = sheet["SIZE"][i]
				elif sheet["BUY_SELL_FLAG"][i] == 1 and sell_min <= sheet["PRICE"][i] * 100 < sell_max:
					if int(sheet["PRICE"][i] * 100 - sell_min) in sell_dict:
						sell_dict[int(sheet["PRICE"][i] * 100 - sell_min)] += sheet["SIZE"][i]
					else:
						sell_dict[int(sheet["PRICE"][i] * 100 - sell_min)] = sheet["SIZE"][i]
			i = i + 1
		else:
			if len(buy_dict) > 0 or len(sell_dict) > 0:
				time_vector.append(time_end)
				buy_vector.append(buy_dict)
				sell_vector.append(sell_dict)
			buy_dict = {}
			sell_dict = {}
			time_start = time_end
			time_end = time_start + datetime.timedelta(microseconds = time_interval * 1000)

	save_orders_json(save_filename, time_vector, buy_vector, sell_vector)

def save_orders_json(save_filename, time_vector, buy_vector, sell_vector):
    order_dict = {"time": time_vector, "buy": buy_vector, "sell": sell_vector}
    df = pd.DataFrame(data=order_dict, columns=["time", "buy", "sell"])
    df.to_json(path_or_buf=save_filename, orient="records", lines=True)
