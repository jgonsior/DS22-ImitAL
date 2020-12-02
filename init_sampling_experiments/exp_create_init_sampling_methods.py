from itertools import combinations
import copy
import multiprocessing
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from numba import jit
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.neural_network import MLPClassifier
import itertools
from active_learning.dataStorage import DataStorage
from active_learning.experiment_setup_lib import init_logger
import sys

# let batches get ordered by different initial_batch_sampling_methodata_storage
# compare that ranking to the optimum
# profit!!
init_logger("console")
#  NR_ITERATIONS = 10000

RANGE_START = int(sys.argv[1])
OUTPUT_DIR = sys.argv[2]


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


def _calculate_uncertainty_metric(batch_indices, data_storage, clf):
    Y_proba = clf.predict_proba(data_storage.X[batch_indices])
    margin = np.partition(-Y_proba, 1, axis=1)
    return np.sum(-np.abs(margin[:, 0] - margin[:, 1]))


def _calculate_randomness_metric(NR_BATCHES, data_storage, clf):
    return np.random.random(NR_BATCHES)


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


def _calculate_predicted_unity(unlabeled_sample_indices, data_storage, clf):
    #  Y_pred = clf.predict(data_storage.X[unlabeled_sample_indices])
    Y_pred = unlabeled_sample_indices
    Y_pred_sorted = sorted(Y_pred)
    count, unique = np.unique(Y_pred_sorted, return_counts=True)
    Y_enc = []
    for i, (c, u) in enumerate(
        sorted(zip(count, unique), key=lambda t: t[1], reverse=True)
    ):
        Y_enc += [i + 1 for _ in range(0, u)]

    Y_enc = np.array(Y_enc)
    counts, unique = np.unique(Y_enc, return_counts=True)
    disagreement_score = sum([c * u for c, u in zip(counts, unique)])
    #  print(Y_pred, "\t -> \t", Y_enc, "\t: ", disagreement_score)
    return disagreement_score


#  print(get_normalized_unity_encoding_mapping(20, 11))
#  exit(-1)

#  BATCH_SIZE = 7
#  for N_CLASSES in range(2, 20):
#      if N_CLASSES >= BATCH_SIZE:
#          N_CLASSES = BATCH_SIZE
#      possible_lengths = set()
#
#      for possible_partition in partitions(BATCH_SIZE):
#          if len(possible_partition) <= N_CLASSES:
#              possible_lengths.add(
#                  sum(
#                      [
#                          c * u
#                          for c, u in zip(
#                              sorted(possible_partition, reverse=True),
#                              range(1, len(possible_partition) + 1),
#                          )
#                      ]
#                  )
#              )
#
#      print(N_CLASSES, ": \t", sorted(possible_lengths))
#  exit(-1)
#
#  N_CLASSES = 10
#  for BATCH_SIZE in range(2, 7):
#  BATCH_SIZE = 7
#  for N_CLASSES in range(2, 20):
#      if N_CLASSES >= BATCH_SIZE:
#          N_CLASSES = BATCH_SIZE
#      lengths = set()
#      for a in np.array(list(itertools.product(range(0, N_CLASSES), repeat=BATCH_SIZE))):
#          lengths.add(_calculate_predicted_unity(a, None, None))
#      print(N_CLASSES, ": \t", lengths)
#  a = _calculate_predicted_unity(np.array([1, 1, 1, 1]), None, None)
#  b = _calculate_predicted_unity(np.array([7, 7, 7, 7]), None, None)
#  assert a == b
#  c = _calculate_predicted_unity(np.array([1, 2, 3, 4]), None, None)
#  assert a < c
#  d = _calculate_predicted_unity(np.array([1, 2, 2, 4]), None, None)
#  assert d < c
#  e = _calculate_predicted_unity(np.array([1, 2, 2, 1]), None, None)
#  assert e < d
#  f = _calculate_predicted_unity(np.array([1, 3, 3, 3]), None, None)
#  assert f < e
#  exit(-1)

for NR_BATCHES in [500, 1000, 250]:
    for RANDOM_SEED in range(RANGE_START * 100, RANGE_START * 100 + 100):
        #  for NR_BATCHES in [50, 100, 250, 500, 1000]:
        #      for RANDOM_SEED in range(RANGE_START * 10000, RANGE_START * 10000 + 10000):
        df = pd.DataFrame(
            [], columns=["source"] + [str(i) for i in range(0, NR_BATCHES)]
        )
        # generate random dataset
        data_storage = DataStorage(
            RANDOM_SEED,
            TEST_FRACTION=0,
            DATASET_NAME="synthetic",
            VARIABLE_DATASET=True,
            NEW_SYNTHETIC_PARAMS=False,
            AMOUNT_OF_FEATURES=100,
            GENERATE_NOISE=True,
            HYPERCUBE=False,
            hyper_parameters={},
            INITIAL_BATCH_SAMPLING_METHOD="furthest",
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

        if NR_BATCHES > len(data_storage.unlabeled_mask):
            possible_batches = data_storage.unlabeled_mask
        else:
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
            _calculate_predicted_unity,
        ]:
            df.loc[len(df.index)] = [str(function) + str(NR_BATCHES)] + [
                function(a, data_storage, clf) for a in possible_batches
            ]

        df.loc[len(df.index)] = [
            "random" + str(NR_BATCHES)
        ] + _calculate_randomness_metric(NR_BATCHES, None, None).tolist()

        if RANDOM_SEED == 0:
            df.to_csv(
                OUTPUT_DIR + "/metric_test_" + str(NR_BATCHES) + ".csv",
                index=False,
                header=True,
            )
        else:
            df.to_csv(
                OUTPUT_DIR + "/metric_test_" + str(NR_BATCHES) + ".csv",
                index=False,
                mode="a",
                header=False,
            )
