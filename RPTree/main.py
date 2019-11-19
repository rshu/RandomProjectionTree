import RPTree
import BuildRPTree
from BuildRPTree import maxminNormalize, para_distance, convertDictToList
from PruneRPTree import PruneTree
import Evaluation
from hyperopt.pyll.stochastic import sample


def main():

    # an improved way is after picking the best pre-processor and learner
    # combination, initialize the input vector with the hyperparameters of
    # both pre-processor and learner
    # the hyperparameters should be normalized first

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

    # need to set a map between normalized list to original dict
    for s in spaceCopy: #TODO
        print(s)
        normalized_s = maxminNormalize(s)
        print(normalized_s)
        print(convertDictToList(normalized_s))
        print("")

    randomSample = BuildRPTree.select(spaceCopy, BuildRPTree.N_SAMPLE)
    # print(len(randomSample))
    root = BuildRPTree.BuildTree(randomSample)
    # print(len(root.key))
    # print(root)
    # print("")
    # print(len(PruneTree(root).key))
    # print(PruneTree(root))

    # Not a working example
    RPResult = 0
    for i in PruneTree(root).key:
        if Evaluation.fitness(i) >= RPResult:
            RPResult = Evaluation.fitness(i)

    print("Random Projection:")
    print("fitness score: ", RPResult) # TODO


if __name__ == "__main__":
    main()
