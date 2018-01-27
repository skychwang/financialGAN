import numpy as np 

def randomOrderGenerator(orders=2, orderSize=10, high=10, low=0):
	while True:
		yield np.random.randint(low, high, size=(orders, 2, orderSize))

#Parameters
"""
orders - the number of orders in a single yielded order stream
orderSize - the size of each order vector within the order stream
high - upperlimit of randomly generated numbers in the order vector
low - lowerlimit of randomly generated numbers in the order vector
"""

#Structure
"""
Sample yield: orders=2, orderSize=10, high=10, low=0

[[[8 6 6 8 9 5 4 2 8 7]
  [9 2 0 1 9 7 6 4 0 1]]

 [[0 3 5 7 1 0 7 5 6 8]
  [5 7 6 6 7 9 6 5 1 8]]]

Structure:
- Order Stream vector
    - Time step 1 vector
        - buy vector
        - sell vector
    - Time step 2 vector
        - buy vector
        - sell vector
    - ...
"""

#Testing
"""
r = randomOrderGenerator()
print(next(r));
"""

