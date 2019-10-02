import random
import matplotlib.pyplot as plt

random.seed(30)


def random_point(dimension=2, min_c=0, max_c=10000):
    return [random.randint(min_c, max_c) for _ in range(dimension)]


def distance(p1=[], p2=[]):
    d = 0
    d += (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    return d


def projection(p=[], polar=[]):
    left = [polar[0], polar[1]]
    right = [polar[2], polar[3]]

    # find point near left or right
    if distance(p, left) <= distance(p, right):
        return 0  # near the left point
    else:
        return 1  # near the right point


def main():
    pop = [random_point() for _ in range(1000)]
    # print(listOfRandomPoint)
    polar = []

    x = [c[0] for c in pop]
    y = [c[1] for c in pop]
    y = [c[1] for c in pop]
    pop = sorted(pop, key=lambda l: l[1], reverse=True)
    randomP = pop[500]
    # print(randomP)

    # find the longest polar
    ds = [distance(i, randomP) for i in pop]
    east = pop[ds.index(max(ds))]
    ds = [distance(i, east) for i in pop]
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
