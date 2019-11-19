import random
from random import randrange
import numpy as np
import math
from sklearn.cluster import KMeans
from hyperopt.pyll.stochastic import sample
from scipy import spatial
from scipy.spatial import distance

import warnings
warnings.filterwarnings("ignore")

import RPTree
import Evaluation

N_POPULATION = 10000
N_DIMENSION = 3
MAX_RANGE = 10000
MIN_RANGE = 0
N_SAMPLE = 1000
BALANCE_THRESHOLD = 1.0
EPSILON = 10.0

random.seed(12345)


def random_point(dimension=N_DIMENSION, min_c=MIN_RANGE, max_c=MAX_RANGE):
    return [random.randint(min_c, max_c) for _, _ in enumerate(range(dimension))]


def sqd(p1, p2):
    return int(math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))))


# compute cosine distance between two hyperparameters
def para_distance(p1, p2):  # TODO

    x = []
    y = []
    for key, value in sorted(p1.items(), key=lambda item: item[0]):
        x.append(value)

    for key, value in sorted(p2.items(), key=lambda item: item[0]):
        y.append(value)

    print("")
    print(x)
    print(y)
    print("")
    print("distance: ", float(distance.euclidean(x, y)))

    return 1 - spatial.distance.cosine(x, y)


def select(data, size):
    res = []
    for _ in range(size):
        if data:
            pos = randrange(len(data))
            elem = data[pos]
            data[pos] = data[-1]
            del data[-1]
            res.append(elem)
    if res:
        return res
    else:
        return None


def balanceValue(left, right):
    if left <= right:  # TODO
        return abs(1 - left / right)
    elif right < left:
        return abs(1 - right / left)
    # else:
    #     return 10 ** 32  # just return a large value


def generatePolar(pop):
    # ----------most distance----------
    # generate N_POLAR**N_POLAR polars
    # however, we want to avoid short distance polars
    polar = {}
    if len(pop) > 0:
        N_POLAR = int(math.log(math.sqrt(len(pop))) + 1)
    else:
        N_POLAR = 1

    for i in range(N_POLAR ** N_POLAR + 100):
        randomP1 = random.choice(pop)
        randomP2 = random.choice(pop)
        polar[i] = {}
        polar[i]['P1'] = randomP1
        polar[i]['P2'] = randomP2
        polar[i]['Distance'] = sqd(randomP1, randomP2)

    sortedPolar = sorted(polar.items(), key=lambda x: x[1]['Distance'])
    sortedPolar.reverse()

    finalPolarList = [sortedPolar[i] for i in range(N_POLAR)]
    finalPolar = dict(finalPolarList)

    return finalPolar


def polarProjection(finalPolar, randomSample):
    projection = {}
    for i in range(len(randomSample)):
        projection[i] = {}
        left = 0
        right = 0
        countPolar = 0

        for key, value in finalPolar.items():
            leftDistance = sqd(randomSample[i], value.get("P1"))
            rightDiatance = sqd(randomSample[i], value.get("P2"))
            currentPolar = "Polar" + str(countPolar)
            if leftDistance <= rightDiatance:
                left += 1
                projection[i][currentPolar] = [0, 1]
            else:
                right += 1
                projection[i][currentPolar] = [1, 0]
            countPolar += 1
            projection[i]["BalanceValue"] = balanceValue(left, right)

    # only keep random sample whose balancevalue is less than threshold
    projectionCopy = {key: value for key, value in projection.copy().items() if
                      value.get("BalanceValue") < BALANCE_THRESHOLD}

    balancedSamples = []

    for k, v in projectionCopy.items():
        balancedSamples.append(randomSample[k])

    # split to eastitems and westitems
    if len(balancedSamples) < 3:
        eastItems = balancedSamples.copy()
        westItems = []
    else:
        balancedSampleArray = np.asarray(balancedSamples)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(balancedSampleArray)
        eastItems = [balancedSamples[i] for i in range(len(balancedSamples)) if kmeans.labels_[i] == 0]
        westItems = [balancedSamples[i] for i in range(len(balancedSamples)) if kmeans.labels_[i] == 1]

    return eastItems, westItems


def BuildTree(population):
    if population is None:
        return None

    # print("Node length:", len(population))
    polars = generatePolar(population)
    eastitems, westitems = polarProjection(polars, population)
    # print("east length:", len(eastitems))
    # print("west length:", len(westitems))
    # print("")

    root = RPTree.TreeNode(population)
    if len(eastitems):
        root.left = BuildTree(eastitems)
    else:
        root.left = None

    if len(westitems):
        root.right = BuildTree(westitems)
    else:
        root.right = None
    return root


# normalize the data with MaxMinScale
# y = (x - min)/(max - min)
def maxminNormalize(x):
    normalzied_x = {}

    x_n_estimators = (x["n_estimators"] - 5) / (50 - 5)
    x_max_depth = (x["max_depth"] - 2) / (100 - 2)
    x_min_samples_split = (x["min_samples_split"] - 0.0) / (1.0 - 0.0)
    x_min_samples_leaf = (x["min_samples_leaf"] - 0.0) / (0.5 - 0.0)
    x_max_leaf_nodes = (x["max_leaf_nodes"] - 2) / (100 - 2)
    x_min_impurity_decrease = (x["min_impurity_decrease"] - 0.0) / (1e-6 - 0.0)
    x_min_weight_fraction_leaf = (x["min_weight_fraction_leaf"] - 0.0) / (0.5 - 0.0)

    normalzied_x["n_estimators"] = x_n_estimators
    normalzied_x["max_depth"] = x_max_depth
    normalzied_x["min_samples_split"] = x_min_samples_split
    normalzied_x["min_samples_leaf"] = x_min_samples_leaf
    normalzied_x["max_leaf_nodes"] = x_max_leaf_nodes
    normalzied_x["min_impurity_decrease"] = x_min_impurity_decrease
    normalzied_x["min_impurity_decrease"] = x_min_weight_fraction_leaf

    return normalzied_x


def convertDictToList(normalized_x):
    res = []

    res.append(normalized_x["n_estimators"])
    res.append(normalized_x["max_depth"])
    res.append(normalized_x["min_samples_split"])
    res.append(normalized_x["min_samples_leaf"])
    res.append(normalized_x["max_leaf_nodes"])
    res.append(normalized_x["min_impurity_decrease"])
    res.append(normalized_x["min_impurity_decrease"])

    return res


def main():
    # pop = [random_point() for _ in range(N_POPULATION)]

    #
    # # generate polars
    # polars = generatePolar(pop)
    # # print(polars)
    #
    # # select random samples from pop
    # randomSample = select(pop.copy(), N_SAMPLE)
    # # print(randomSample)
    #
    # # project selected random samples to polars
    # eastitems, westitems = polarProjection(polars, randomSample)
    # print("eastitems:")
    # print(eastitems)
    # print("westitems:")
    # print(westitems)

    # Build Random Projection Tree
    # print(BuildTree(pop))

    # root = BuildTree(pop)
    # print(root.left)
    # print(root.right)

    x = sample(Evaluation.para_space)
    print("x: ", x)
    y = sample(Evaluation.para_space)
    print("y: ", y)

    normalized_x = maxminNormalize(x)
    normalized_y = maxminNormalize(y)
    print("normalized x: ", normalized_x)
    print("normalized y: ", normalized_y)

    print("Before max min scaler:")
    print(para_distance(x, y))

    print("After max min scaler: ")
    print(para_distance(normalized_x, normalized_y))

    space = [sample(Evaluation.para_space) for _ in range(50)]
    # print(space)

    # make a copy of parameter space
    spaceCopy = space.copy()
    # print(spaceCopy)

    for s in spaceCopy:
        print(s)
        normalized_s = maxminNormalize(s)
        print(normalized_s)
        print(convertDictToList(normalized_s))
        print("")

# if __name__ == "__main__":
#     main()
