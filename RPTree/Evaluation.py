import numpy as np


# maximize fitness function
def fitness(s):
    squresum = sum(map(lambda x: x ** 3, s))
    ssum = sum(map(lambda x: x, s))
    return np.sqrt(squresum) / ssum


def main():
    s = [2222, 3333, 4444]
    print(fitness(s))


if __name__ == "__main__":
    main()
