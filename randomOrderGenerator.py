import numpy as np 

def randomOrderGenerator(numGenerate=1000, orderLength=50, orderStreamSize=100, high=101, low=0):
	while True:
		yield np.random.randint(low, high, size=(numGenerate, orderStreamSize, orderLength, 1))

def randomLabelGenerator(numGenerate=1000):
  while True:
    yield np.random.randint(0, 2, size=(numGenerate))

#Testing
#r = randomOrderGenerator(2, 5, 3)
#print(next(r))
#l = randomLabelGenerator()
#print(next(l))

