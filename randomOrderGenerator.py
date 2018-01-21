import numpy as np 

def randomOrderGenerator(orders=2, orderSize=10, high=10, low=0):
	while True:
		yield np.random.randint(low, high, size=(orders, orderSize))

#Testing
"""
r = randomOrderGenerator()
print(next(r));
print(next(r));
print(next(r));
"""
