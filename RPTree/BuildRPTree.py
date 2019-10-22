import random
from random import randrange
import numpy as np
import math
from sklearn.cluster import KMeans

import RPTree

N_POPULATION = 50000
N_DIMENSION = 3
MAX_RANGE = 10000
MIN_RANGE = 0
BALANCE_THRESHOLD = 1.0

random.seed(12345)


def random_point(dimension=N_DIMENSION, min_c=MIN_RANGE, max_c=MAX_RANGE):
    return [random.randint(min_c, max_c) for _, _ in enumerate(range(dimension))]


def sqd(p1, p2):
    return int(math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))))


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
    if left <= right:
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
    N_POLAR = int(math.log(math.sqrt(len(pop))) + 1)

    for i in range(N_POLAR ** N_POLAR):  # how to sample
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

    projectionCopy = {key: value for key, value in projection.copy().items() if
                      value.get("BalanceValue") < BALANCE_THRESHOLD}

    balancedSamples = []

    for k, v in projectionCopy.items():
        balancedSamples.append(randomSample[k])

    if len(balancedSamples) < 3:
        eastItems = balancedSamples.copy()
        westItems = []
    else:
        balancedSampleArray = np.asarray(balancedSamples)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(balancedSampleArray)
        eastItems = [balancedSamples[i] for i in range(len(balancedSamples)) if kmeans.labels_[i] == 0]
        westItems = [balancedSamples[i] for i in range(len(balancedSamples)) if kmeans.labels_[i] == 1]

    return eastItems, westItems


def BuildTree(RandomSample):
    if RandomSample is None:
        return None

    root = RPTree.TreeNode(RandomSample)
    root.left = BuildTree(polarProjection(RandomSample)[0])
    root.right = BuildTree(polarProjection(RandomSample)[1])
    return root




def main():
    pop = [random_point() for _ in range(N_POPULATION)]

    # Build Random Projection Tree
    BuildTree(pop)


    # Prune Tree Nodes




if __name__ == "__main__":
    main()
