import numpy as np
from hyperopt import hp


# maximize fitness function
def fitness(s):
    squresum = sum(map(lambda x: x ** 3, s))
    ssum = sum(map(lambda x: x, s))
    return np.sqrt(squresum) / ssum


space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}


para_space = {
    'n_estimators' : hp.quniform('n_estimators', 10, 150, 1),
    'max_depth': hp.quniform('max_depth', 2, 100, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_leaf_nodes': hp.quniform('max_leaf_nodes', 2, 100, 1),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 1e-6),
    'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 1.0)
}





def main():
    s = [2222, 3333, 4444]
    print(fitness(s))


if __name__ == "__main__":
    main()
