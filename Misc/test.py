import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from collections import OrderedDict
from operator import getitem
import math

# parameter initialization
N_POPULATION = 500
N_DIMENSION = 3
N_POLAR = int(math.log(math.sqrt(N_POPULATION)) + 1)
MAX_RANGE = 10000
MIN_RANGE = 0
random.seed(12345)


def random_point(dimension=N_DIMENSION, min_c=MIN_RANGE, max_c=MAX_RANGE):
    return [random.randint(min_c, max_c) for _, _ in enumerate(range(dimension))]


def sqd(p1, p2):
    return int(math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))))


def projection(p=[], polar=[]):
    left = [polar[0], polar[1]]
    right = [polar[2], polar[3]]

    # find point near left or right
    if sqd(p, left) <= sqd(p, right):
        return 0  # near the left point
    else:
        return 1  # near the right point


def RPTree():
    pass


def main():
    # initialize population with random points
    pop = [random_point() for _ in range(N_POPULATION)]
    # print(pop)
    xs = [p[0] for p in pop]
    ys = [p[1] for p in pop]
    zs = [p[2] for p in pop]

    # plot pop in 3D space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, marker='o')
    plt.title('Random population in 3D space')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # plt.show()

    # generate N_POLAR**N_POLAR polars
    # however, we want to avoid short polars
    polar = {}
    print(N_POLAR ** N_POLAR)

    for i in range(N_POLAR ** N_POLAR):  # how to sample
        randomP1 = random.choice(pop)
        randomP2 = random.choice(pop)
        polar[i] = {}
        polar[i]['P1'] = randomP1
        polar[i]['P2'] = randomP2
        polar[i]['Distance'] = sqd(randomP1, randomP2)

    # print("polar", polar)
    sortedPolar = sorted(polar.items(), key=lambda x: x[1]['Distance'])
    sortedPolar.reverse()
    # print("sortedpolar", sortedPolar)

    # only keep top N_POLAR polars
    finalPolarList = [sortedPolar[i] for i in range(N_POLAR)]
    finalPolar = dict(finalPolarList)  # convert list to dict
    print(finalPolar)

    for key1, value1 in finalPolar.items():
        print(value1)
        p1 = value1.get("P1")
        p2 = value1.get("P2")
        plt.plot(p1, p2, marker='.', linewidth=2)

    # find the longest line
    pop.sort(key=lambda x: x[0])
    midP = pop[int(N_POPULATION / 2)]

    ds = [sqd(i, midP) for i in pop]
    east = pop[ds.index(max(ds))]
    ds = [sqd(i, east) for i in pop]
    west = pop[ds.index(max(ds))]

    print("east: ", east, "west:", west, "Distance: ", sqd(east, west))
    plt.plot(east, west, 'k^-')
    plt.show()

    exit(1)

    polar = []

    x = [c[0] for c in pop]
    y = [c[1] for c in pop]
    pop = sorted(pop, key=lambda l: l[1], reverse=True)
    randomP = pop[500]
    # print(randomP)

    # find the longest polar
    ds = [sqd(i, randomP) for i in pop]
    east = pop[ds.index(max(ds))]
    ds = [sqd(i, east) for i in pop]
    west = pop[ds.index(max(ds))]

    if east[0] < west[0]:
        polar.append(east + west)
    else:
        polar.append(west + east)

    # generate another 4 polars
    for _ in range(4):
        p1, p2 = random.sample(pop, 2)
        if p1[0] < p2[0]:
            polar.append(p1 + p2)
        else:
            polar.append(p2 + p1)
    print("All polars:", polar)

    plt.scatter(x, y, label="point", color="green", marker=".", s=30)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Plot random points and polars')
    for i in range(len(polar)):
        # print(polar[i])
        p1 = [polar[i][0], polar[i][1]]
        p2 = [polar[i][2], polar[i][3]]
        plt.plot(p1, p2, 'ro-')
    # plt.plot(east, west, 'ro-')
    plt.legend()
    plt.show()

    # label = {}
    label = []
    for i, x in enumerate(pop):
        cell = []
        for j, p in enumerate(polar):
            # print(x, p, projection(x, p))
            cell.append(projection(x, p))
            # label[i, j] = projection(x, p)
        label.append(cell)

    print(label)


if __name__ == "__main__":
    main()
