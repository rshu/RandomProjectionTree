import RPTree
import BuildRPTree
from BuildRPTree import maxminNormalize, para_distance, convertDictToList
from BuildRPTree import convertListToDict, reverse_minmaxScaler
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

    print("")
    print("Before max min scaler:")
    print(para_distance(x, y))

    print("")
    print("After max min scaler: ")
    print(para_distance(normalized_x, normalized_y))
    print("")

    space = [sample(Evaluation.para_space) for _ in range(50)]
    # print(space)

    # make a copy of parameter space
    spaceCopy = space.copy()
    # print(spaceCopy)

    # need to set a map between normalized list to original dict
    for s in spaceCopy:
        print(s)
        normalized_s = maxminNormalize(s)
        print(normalized_s)
        list = convertDictToList(normalized_s)
        print(list)
        dict = convertListToDict(list)
        print(dict)
        r_dict = reverse_minmaxScaler(dict)
        print(r_dict)
        print("")

    randomSample = BuildRPTree.select(spaceCopy, BuildRPTree.N_SAMPLE)
    # print(len(randomSample))
    root = BuildRPTree.BuildTree(randomSample) # issues with building the rp tree
    print(len(root.key))
    print(root)
    print(root.left)
    print(root.right)
    print("")
    print(len(PruneTree(root).key))
    print(PruneTree(root))
    exit(1)


    # Not a working example
    RPResult = 0
    for i in PruneTree(root).key:
        if Evaluation.fitness(i) >= RPResult:
            RPResult = Evaluation.fitness(i)

    print("Random Projection:")
    print("fitness score: ", RPResult) # TODO

# need to address both numeric and categorical hyperparameters
# before Dec 10th
# check GAN for security

if __name__ == "__main__":
    main()
