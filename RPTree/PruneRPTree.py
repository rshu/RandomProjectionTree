import RPTree
import BuildRPTree
import random
import Evaluation

random.seed(5000)

def PruneTree(root):
    if root is None:
        return

    # print(root)
    # print(len(root.key))

    PruneTree(root.left)
    PruneTree(root.right)

    if root.left is None and root.right is None:
        return root
    elif root.right is None and root.left is not None:
        root.key = root.left.key.copy()
        return root
    elif root.left is None and root.right is not None:
        root.key = root.right.key.copy()
        return root
    elif root.left is not None and root.right is not None:
        l = random.choice(root.left.key)
        r = random.choice(root.right.key)

        evaluationLeft = Evaluation.fitness(l)
        evaluationRight = Evaluation.fitness(r)

        if evaluationLeft - evaluationRight >= BuildRPTree.EPSILON:
            root.key = root.left.key.copy()
        elif evaluationRight - evaluationLeft > BuildRPTree.EPSILON:
            root.key = root.right.key.copy()
        return root


def main():
    pop = [BuildRPTree.random_point() for _ in range(BuildRPTree.N_POPULATION)]

    gridresult = []
    randomresult = []

    # grid search
    for sample in pop:
        gridresult.append(Evaluation.fitness(sample))

    print("Grid search:")
    # print(gridresult)
    print(max(gridresult))
    print("")

    # random search
    randomSample = BuildRPTree.select(pop.copy(), 100)

    for sample in randomSample:
        randomresult.append(Evaluation.fitness(sample))

    print("Random search:")
    # print(randomresult)
    print(max(randomresult))
    print("")


    root = BuildRPTree.BuildTree(pop)
    # print(len(root.key))
    # print(root)
    # print("")
    # print(len(PruneTree(root).key))
    # print(PruneTree(root))

    RPResult = 0
    for i in PruneTree(root).key:
        if Evaluation.fitness(i) >= RPResult:
            RPResult = Evaluation.fitness(i)

    print("Random Projection:")
    print(RPResult)




if __name__ == "__main__":
    main()
