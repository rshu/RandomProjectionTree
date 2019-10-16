import test
import numpy as np



# maximize fitness function
def fitness(s):
    squresum = sum(map(lambda x: x ** 3, s))
    ssum = sum(map(lambda x: x, s))
    return np.sqrt(squresum) / ssum


def main():
    pop = [test.random_point() for _ in range(test.N_POPULATION)]
    # print(len(pop))

    gridresult = []
    randomresult = []

    # grid search
    for sample in pop:
        gridresult.append(fitness(sample))

    print("Grid search:")
    # print(gridresult)
    print(max(gridresult))

    # random search
    randomSample = test.select(pop.copy(), 100)

    for sample in randomSample:
        randomresult.append(fitness(sample))

    print("Random search:")
    # print(randomresult)
    print(max(randomresult))

    print("Random Projection Tree:")
    print("")

    test.RPTree(pop, 0, None)
    # print(test.OutputRPTree)
    test.bottomUp(test.OutputRPTree)



if __name__ == "__main__":
    main()
