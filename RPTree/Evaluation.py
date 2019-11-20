import numpy as np
from hyperopt import hp
import pandas as pd
from hyperopt.pyll.stochastic import sample
from sklearn.ensemble import RandomForestClassifier
import random
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

import warnings

warnings.filterwarnings("ignore")


# maximize fitness function
def fitness(s):
    squresum = sum(map(lambda x: x ** 3, s))
    ssum = sum(map(lambda x: x, s))
    return np.sqrt(squresum) / ssum


def _define_smote(data, num, k=5, r=1):
    corpus = []
    if len(data) < k:
        k = len(data) - 1
    nbrs = NearestNeighbors(
        n_neighbors=k, algorithm='ball_tree', p=r).fit(data)
    _, indices = nbrs.kneighbors(data)
    for _ in range(0, num):
        mid = random.randint(0, len(data) - 1)
        nn = indices[mid, random.randint(1, k - 1)]
        datamade = []
        for j in range(0, len(data[mid])):
            gap = random.random()
            datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])
        corpus.append(datamade)
    corpus = np.array(corpus)
    corpus = np.vstack((corpus, np.array(data)))
    return corpus


def SMOTE(train_df):
    m, k, r = 264, 10, 5
    X = train_df.loc[:, train_df.columns != 'label']
    y = train_df.label
    pos_train, neg_train = X.loc[y == 1], X.loc[y == 0]

    pos_train = _define_smote(pos_train.values, m, k, r)
    neg_train = neg_train.sample(min(m, neg_train.shape[0])).values
    X = np.concatenate((pos_train, neg_train), axis=0)
    y = [1] * pos_train.shape[0] + [0] * neg_train.shape[0]
    y = np.asarray(y).reshape(-1, 1)
    balanced = pd.DataFrame(
        np.concatenate((X, y), axis=1), columns=train_df.columns).astype('int')
    return balanced


# param_grid = {
#     'class_weight': [None, 'balanced'],
#     'boosting_type': ['gbdt', 'goss', 'dart'],
#     'num_leaves': list(range(30, 150)),
#     'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=1000)),
#     'subsample_for_bin': list(range(20000, 300000, 20000)),
#     'min_child_samples': list(range(20, 500, 5)),
#     'reg_alpha': list(np.linspace(0, 1)),
#     'reg_lambda': list(np.linspace(0, 1)),
#     'colsample_bytree': list(np.linspace(0.6, 1, 10))
# }
#
# # Subsampling (only applicable with 'goss')
# subsample_dist = list(np.linspace(0.5, 1, 100))

# para_space = {
#     'n_estimators': hp.choice('n_estimators', range(50, 150)),
#     'max_depth': hp.quniform('max_depth', 2, 100, 1),
#     'min_samples_split': hp.uniform('min_samples_split', 0.0, 1.0),
#     'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
#     'max_leaf_nodes': hp.choice('max_leaf_nodes', range(2, 100)),
#     'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 1e-6),
#     'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 0.5)
# }

para_space = {
    'n_estimators': hp.choice('n_estimators', range(50, 150)),
    'max_depth': hp.quniform('max_depth', 2, 100, 1),
    'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(2, 10)),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', range(2, 50)),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 1e-6),
    'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 0.5)
}


def read_data(dataset):
    train_df = pd.read_csv(f'../data/FARSEC/{dataset}-train.csv').drop(
        ['id'], axis=1)
    test_df = pd.read_csv(f'../data/FARSEC/{dataset}-test.csv').drop(
        ['id'], axis=1)

    return train_df, test_df


def apply_model(train_df, test_df, model):
    X = train_df.loc[:, train_df.columns != 'label']
    y = train_df.label
    model.fit(X, y)
    X_test = test_df.loc[:, test_df.columns != 'label']
    prediction = model.predict(X_test)
    return prediction


def get_prediction(test_labels, prediction):
    tn, fp, fn, tp = confusion_matrix(
        test_labels, prediction, labels=[0, 1]).ravel()
    pre = 1.0 * tp / (tp + fp) if (tp + fp) != 0 else 0
    rec = 1.0 * tp / (tp + fn) if (tp + fn) != 0 else 0
    spec = 1.0 * tn / (tn + fp) if (tn + fp) != 0 else 0
    fpr = 1 - spec
    npv = 1.0 * tn / (tn + fn) if (tn + fn) != 0 else 0
    acc = 1.0 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) != 0 else 0
    gm = 2.0 * rec * (1 - fpr) / (rec + 1 - fpr) if (rec + 1 - fpr) != 0 else 0

    print("Recall: ", rec)
    print("False positive rate: ", fpr)
    print("Precision: ", pre)
    print("F-measure: ", f1)
    print("G-measure: ", gm)

    return rec, fpr, pre, f1, gm


# def LGB(params):
#     class_weight = params["class_weight"]
#     boosting_type = params["boosting_type"]
#     num_leaves = params["num_leaves"]
#     learning_rate = params["learning_rate"]
#     subsample_for_bin = params["subsample_for_bin"]
#     min_child_samples = params["min_child_samples"]
#     reg_alpha = params["reg_alpha"]
#     reg_lambda = params["reg_lambda"]
#     colsample_bytree = params["colsample_bytree"]
#     subsample = params["subsample"]
#     model = lgb.LGBMClassifier(class_weight=class_weight, boosting_type=boosting_type,
#                                num_leaves=num_leaves, learning_rate=learning_rate,
#                                subsample_for_bin=subsample_for_bin, min_child_samples=min_child_samples,
#                                reg_alpha=reg_alpha, reg_lambda=reg_lambda, colsample_bytree=colsample_bytree,
#                                subsample=subsample)
#     return model


def RF(params):
    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]
    max_leaf_nodes = params["max_leaf_nodes"]
    min_impurity_decrease = params["min_impurity_decrease"]
    min_samples_leaf = params["min_samples_leaf"]
    min_samples_split = params["min_samples_split"]
    min_weight_fraction_leaf = params["min_weight_fraction_leaf"]
    # model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,
    #                                min_impurity_decrease=min_impurity_decrease, min_samples_leaf=min_samples_leaf,
    #                                min_samples_split=min_samples_split,
    #                                min_weight_fraction_leaf=min_weight_fraction_leaf)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        min_weight_fraction_leaf=min_weight_fraction_leaf
    )
    return model


def fitness_RF(params):

    train_df, test_df = read_data("ambari-clni")
    train_df = SMOTE(train_df)
    X = train_df.loc[:, train_df.columns != 'label']
    y = train_df.label
    X_test = test_df.loc[:, test_df.columns != 'label']
    test_labels = test_df.label.values.tolist()

    model = RF(params)
    model.fit(X, y)
    prediction = model.predict(X_test)
    rec, fpr, pre, f1, gm = get_prediction(test_labels, prediction)

    return gm


def main():
    params = sample(para_space)
    # params['subsample'] = random.sample(subsample_dist, 1)[0] if params['boosting_type'] != 'goss' else 1.0
    print(params)

    train_df, test_df = read_data("ambari-clni")
    train_df = SMOTE(train_df)
    X = train_df.loc[:, train_df.columns != 'label']
    y = train_df.label
    X_test = test_df.loc[:, test_df.columns != 'label']
    test_labels = test_df.label.values.tolist()

    model = RF(params)
    # model = RandomForestClassifier(random_state=15325)
    # model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                    max_depth=95.0, max_features='auto', max_leaf_nodes=5,
    #                    min_impurity_decrease=1.4453418778034065e-07,
    #                    min_impurity_split=None, min_samples_leaf=6,
    #                    min_samples_split=8,
    #                    min_weight_fraction_leaf=0.06271981064352494,
    #                    n_estimators=77, n_jobs=None, oob_score=False,
    #                    random_state=None, verbose=0, warm_start=False)
    # model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                    max_depth=2.0, max_features='auto', max_leaf_nodes=27,
    #                    min_impurity_decrease=6.255259369350126e-07,
    #                    min_impurity_split=None, min_samples_leaf=8,
    #                    min_samples_split=2,
    #                    min_weight_fraction_leaf=0.007253257758598586,
    #                    n_estimators=80, n_jobs=None, oob_score=False,
    #                    random_state=None, verbose=0, warm_start=False)
    print(model)
    model.fit(X, y)
    prediction = model.predict(X_test)
    # print(prediction)
    # exit(1)

    get_prediction(test_labels, prediction)


if __name__ == "__main__":
    main()
