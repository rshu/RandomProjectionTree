import numpy as np


# maximize fitness function
def fitness(s):
    squresum = sum(map(lambda x: x ** 3, s))
    ssum = sum(map(lambda x: x, s))
    return np.sqrt(squresum) / ssum

