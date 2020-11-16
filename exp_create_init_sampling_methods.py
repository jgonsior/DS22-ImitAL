import copy
from joblib import Parallel, delayed, parallel_backend
from numba import jit
import random
import math
import pandas as pd
import numpy as np
import timeit
from sklearn.datasets import make_classification
from active_learning.dataStorage import DataStorage
from active_learning.experiment_setup_lib import init_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import pairwise_distances, accuracy_score
import multiprocessing
from active_learning.sampling_strategies.learnedBaseBatchSampling import (
    LearnedBaseBatchSampling,
)

# let batches get ordered by different initial_batch_sampling_methodata_storage
# compare that ranking to the optimum
# profit!!
init_logger("console")
NR_ITERATIONS = 1


@jit(nopython=True)
def _find_firsts(items, vec):
    """return the index of the first occurence of item in vec"""
    result = []
    for item in items:
        for i in range(len(vec)):
            if item == vec[i]:
                result.append(i)
                break
    return result


def _calculate_furthest_metric(batch_indices, data_storage, clf):
    return np.sum(pairwise_distances(data_storage.X[batch_indices]))


def _calculate_furthest_lab_metric(batch_indices, data_storage, clf):
    return np.sum(
        pairwise_distances(
            data_storage.X[batch_indices],
            data_storage.X[data_storage.labeled_mask],
        )
    )


def _calculate_graph_density_metric(batch_indices, data_storage, clf):
    return np.sum(
        data_storage.graph_density[
            _find_firsts(batch_indices, data_storage.initial_unlabeled_mask)
        ]
    )


def _calculate_uncertainty_metric(batch_indices, data_storage, clf):
    Y_proba = clf.predict_proba(data_storage.X[batch_indices])
    margin = np.partition(-Y_proba, 1, axis=1)
    return np.sum(-np.abs(margin[:, 0] - margin[:, 1]))


def _future_peak(unlabeled_sample_indices, data_storage, clf):
    copy_of_classifier = copy.deepcopy(clf)

    copy_of_labeled_mask = np.append(
        data_storage.labeled_mask, unlabeled_sample_indices, axis=0
    )

    copy_of_classifier.fit(
        data_storage.X[copy_of_labeled_mask], data_storage.Y[copy_of_labeled_mask]
    )

    Y_pred_test = copy_of_classifier.predict(data_storage.X)
    Y_true = data_storage.Y
    return accuracy_score(Y_pred_test, Y_true)


for NR_BATCHES in [50, 100, 250, 500, 1000]:
    df = pd.DataFrame([], columns=["source"] + [str(i) for i in range(0, NR_BATCHES)])
    for RANDOM_SEED in range(0, NR_ITERATIONS):
        # generate random dataset
        data_storage = DataStorage(
            RANDOM_SEED,
            TEST_FRACTION=0,
            DATASET_NAME="synthetic",
            VARIABLE_DATASET=False,
            NEW_SYNTHETIC_PARAMS=False,
            AMOUNT_OF_FEATURES=100,
            GENERATE_NOISE=True,
            HYPERCUBE=False,
            hyper_parameters={},
            INITIAL_BATCH_SAMPLING_METHOD="graph_density",
        )

        # generate some pre-existent labels
        amount_of_prelabeled = random.randint(1, 50)
        random_labeled_samples_indices = np.random.choice(
            data_storage.unlabeled_mask, amount_of_prelabeled, replace=False
        )
        data_storage.label_samples(
            random_labeled_samples_indices,
            data_storage.Y[random_labeled_samples_indices],
            "I",
        )

        # sneak peak into future for n batches -> optimum ranking
        clf = MLPClassifier(verbose=0)
        clf.fit(
            data_storage.X[data_storage.labeled_mask],
            data_storage.Y[data_storage.labeled_mask],
        )

        possible_batches = [
            np.random.choice(
                data_storage.unlabeled_mask,
                size=NR_BATCHES,
                replace=False,
            )
            for x in range(0, NR_BATCHES)
        ]

        with parallel_backend("loky", n_jobs=multiprocessing.cpu_count()):
            future_peak_accs = Parallel()(
                delayed(_future_peak)(
                    unlabeled_sample_index,
                    data_storage,
                    clf,
                )
                for unlabeled_sample_index in possible_batches
            )

        df.loc[len(df.index)] = ["future"] + future_peak_accs
        for function in [
            _calculate_furthest_metric,
            _calculate_uncertainty_metric,
            _calculate_furthest_lab_metric,
            _calculate_graph_density_metric,
        ]:
            df.loc[len(df.index)] = [str(function) + str(NR_BATCHES)] + [
                function(a, data_storage, clf) for a in possible_batches
            ]

        #  print(df)
        if RANDOM_SEED == 0:
            df.to_csv(
                "metric_test_" + str(NR_BATCHES) + ".csv", index=False, header=True
            )
        else:
            df.to_csv(
                "metric_test_" + str(NR_BATCHES) + ".csv",
                index=False,
                mode="a",
                header=False,
            )
