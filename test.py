import math
import pandas as pd
import numpy as np
import timeit
from sklearn.datasets import make_classification

n_samples = 1000

X, Y = make_classification(n_samples=n_samples, n_features=20)


def pandas():
    df = pd.DataFrame(X)

    labeled_X = df[math.floor(n_samples / 2) :]
    unlabeled_X = df[: math.floor(n_samples / 2)]

    unlabeled_Y = pd.DataFrame(data=Y[math.floor(n_samples / 2) :], columns=["label"])
    labeled_Y = pd.DataFrame(data=Y[: math.floor(n_samples / 2)], columns=["label"])

    for i in range(0, 50):
        query_indices = unlabeled_X.sample(5).index
        Y_query = [1, 2, 3, 4, 5]
        labeled_X = labeled_X.append(unlabeled_X.loc[query_indices])
        labeled_Y = labeled_Y.append(Y_query)
        unlabeled_X = unlabeled_X.drop(query_indices)
        unlabeled_Y = unlabeled_Y.drop(query_indices)

        for i in range(0, 5):
            mean = labeled_X.mean()
            mean += labeled_Y.mean()
            mean += unlabeled_Y.mean()
            mean += unlabeled_Y.mean()
        print(mean)


def numpy():
    labeled_mask = np.arange(math.floor(n_samples / 2), n_samples)
    unlabeled_mask = np.arange(0, math.floor(n_samples / 2))

    for i in range(0, 50):
        query_indices = np.random.choice(unlabeled_mask, 5)
        Y_query = [1, 2, 3, 4, 5]

        labeled_mask = np.append(labeled_mask, query_indices, axis=0)
        unlabeled_mask = np.delete(unlabeled_mask, query_indices)

        Y[query_indices] = Y_query

        for i in range(0, 5):
            mean = np.mean(X[labeled_mask])
            mean += np.mean(Y[labeled_mask])
            mean += np.mean(X[unlabeled_mask])
            mean += np.mean(Y[unlabeled_mask])
        print(mean)


n_iter = 10

np_time = timeit.timeit(lambda: numpy(), number=n_iter)
pd_time = timeit.timeit(lambda: pandas(), number=n_iter)
print(pd_time)
print(np_time)
