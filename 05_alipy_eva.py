from active_learning.query_sampling_strategies.TrainedImitALQuerySampler import (
    TrainedImitALBatchSampler,
    TrainedImitALSampler,
    TrainedImitALSingleSampler,
)
from active_learning.dataStorage import DataStorage
from active_learning.datasets.uci import load_uci
from active_learning.datasets.dwtc import load_dwtc
from active_learning.datasets.synthetic import load_synthetic
import argparse
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
import json
import csv
import math
import multiprocessing
import os
import random
import sys
from timeit import default_timer as timer
from operator import itemgetter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from alipy.data_manipulate.al_split import split
from alipy.experiment.al_experiment import AlExperiment
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, auc
from imitLearningPipelineSharedCode import non_slurm_strategy_ids, dataset_id_mapping
from active_learning.logger.logger import init_logger

init_logger("tmp.log")

parser = argparse.ArgumentParser()
parser.add_argument("--DATASETS_PATH", default="../datasets")
parser.add_argument("--N_JOBS", type=int, default=1)
parser.add_argument(
    "--INDEX", type=int, default=1, help="Specifies which dataset to use etc."
)
parser.add_argument("--OUTPUT_PATH", default="../datasets/ali")
parser.add_argument("--RANDOM_SEEDS_INPUT_FILE")

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()


# first open job file to get the real random_seed from it
random_seed_df = pd.read_csv(
    config.RANDOM_SEEDS_INPUT_FILE,
    header=0,
    index_col=0,
    nrows=config.INDEX + 1,
)
DATASET_ID, STRATEGY_ID, DATASET_RANDOM_SEED = random_seed_df.loc[config.INDEX]


np.random.seed(DATASET_RANDOM_SEED)
random.seed(DATASET_RANDOM_SEED)
BATCH_SIZE = 5
DATASET_NAME = dataset_id_mapping[DATASET_ID][0]

# specify which dataset to load
print("dataset: ", DATASET_NAME)
print("random_seed: ", DATASET_RANDOM_SEED)

if DATASET_NAME == "synthetic":
    df, synthetic_creation_args = load_synthetic(
        DATASET_RANDOM_SEED,
        NEW_SYNTHETIC_PARAMS=True,
        VARIABLE_DATASET=True,
        AMOUNT_OF_FEATURES=20,
        HYPERCUBE=True,
        GENERATE_NOISE=True,
    )
elif DATASET_NAME == "dwtc":
    df, synthetic_creation_args = load_dwtc(
        DATASETS_PATH=config.DATASETS_PATH, RANDOM_SEED=DATASET_RANDOM_SEED
    )
else:
    df, synthetic_creation_args = load_uci(
        DATASETS_PATH=config.DATASETS_PATH,
        RANDOM_SEED=DATASET_RANDOM_SEED,
        DATASET_NAME=DATASET_NAME,
    )

data_storage = DataStorage(df, TEST_FRACTION=0)
X = data_storage.X
Y = data_storage.exp_Y


class ALiPYImitALSingle:
    trained_imitAL_sampler: TrainedImitALSampler

    def __init__(self, X=None, Y=None, **kwargs):
        # load NN params from json file
        with open(
            os.path.dirname(kwargs["NN_BINARY_PATH"])
            + "/01_dataset_creation_stats.csv_params.json",
            "r",
        ) as f:
            content = str(f.read())
            dataset_stats = json.loads(content)

        print(dataset_stats)

        self.trained_imitAL_sampler = TrainedImitALSingleSampler(
            PRE_SAMPLING_METHOD=dataset_stats["INITIAL_BATCH_SAMPLING_METHOD"],
            PRE_SAMPLING_ARG=dataset_stats["INITIAL_BATCH_SAMPLING_ARG"],
            AMOUNT_OF_PEAKED_OBJECTS=dataset_stats["AMOUNT_OF_PEAKED_OBJECTS"],
            DISTANCE_METRIC=dataset_stats["DISTANCE_METRIC"],
            STATE_ARGSECOND_PROBAS=dataset_stats["STATE_ARGSECOND_PROBAS"],
            STATE_ARGTHIRD_PROBAS=dataset_stats["STATE_ARGTHIRD_PROBAS"],
            STATE_DIFF_PROBAS=dataset_stats["STATE_DIFF_PROBAS"],
            STATE_PREDICTED_CLASS=dataset_stats["STATE_PREDICTED_CLASS"],
            STATE_DISTANCES_LAB=dataset_stats["STATE_DISTANCES_LAB"],
            STATE_DISTANCES_UNLAB=dataset_stats["STATE_DISTANCES_UNLAB"],
            STATE_INCLUDE_NR_FEATURES=dataset_stats["STATE_INCLUDE_NR_FEATURES"],
            NN_BINARY_PATH=kwargs["NN_BINARY_PATH"],
        )

    def select(self, labeled_index, unlabeled_index, model, batch_size=1, **kwargs):
        labeled_index = self.X[labeled_index]

        max_sum = 0

        for i in range(0, kwargs["HYPER_SAMPLING_SEARCH_ITERATIONS"]):

            # if there are less than 20 samples left we  still have to show the ann 20 samples because of the fixed input size
            # so we just "pad" it with random indices
            if len(unlabeled_index) <= 20:
                padding = np.random.choice(unlabeled_index, 20 - len(unlabeled_index))
                possible_samples_indices = np.concatenate((unlabeled_index, padding))
                break
            random_sample_index = unlabeled_index.random_sample(
                20
            )  # self._fast_random_choice_list(unlabeled_index, 20)

            random_sample = self.X[random_sample_index]
            # calculate distance to each other
            total_distance = np.sum(
                self._pairwise_distances_subsampling(random_sample, random_sample)
            )

            total_distance += np.sum(
                self._pairwise_distances_subsampling(labeled_index, random_sample)
            )
            if total_distance > max_sum:
                max_sum = total_distance
                possible_samples_indices = random_sample_index

        X_query = self.X[possible_samples_indices]

        X_state = self.calculate_state(
            X_query,
            STATE_ARGSECOND_PROBAS=True,
            STATE_DIFF_PROBAS=False,
            STATE_ARGTHIRD_PROBAS=True,
            STATE_DISTANCES_LAB=True,
            STATE_DISTANCES_UNLAB=True,
            STATE_PREDICTED_CLASS=False,
            model=model,
            labeled_index=labeled_index,
            train_unlabeled_X=self.X[unlabeled_index],
            TRUE_DISTANCES=kwargs["TRUE_DISTANCES"],
        )
        X_state = np.reshape(X_state, (1, len(X_state)))
        Y_pred = self.sampling_classifier.predict(X_state)
        sorting = Y_pred
        # use the optimal values
        zero_to_one_values_and_index = list(zip(sorting, possible_samples_indices))
        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        return [v for k, v in ordered_list_of_possible_sample_indices[:batch_size]]


test = ALiPYImitALSingle(
    X=X, Y=Y, NN_BINARY_PATH=config.OUTPUT_PATH + "/03_imital_trained_ann.model"
)
exit(-1)


class ANNQuerySingle:
    def __init__(self, X=None, Y=None, **kwargs):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        with open(kwargs["NN_BINARY_PATH"], "rb") as handle:
            model = dill.load(handle)

        self.sampling_classifier = model

        self.X = X
        self.Y = Y
        self.DISTANCE_METRIC = kwargs["DISTANCE_METRIC"]
        self.STATE_INCLUDE_NR_FEATURES = kwargs["STATE_INCLUDE_NR_FEATURES"]

    def _fast_random_choice_numpy(self, elements, size):
        return elements[random.sample(range(0, len(elements)), size)]

    def _pairwise_distances_subsampling(self, X, Y, sampling_threshold=100):
        # subsample X or Y for less computational cost
        # if len(X) > sampling_threshold:
        #    rand_idx = self._fast_random_choice_numpy(
        #        np.arange(0, np.shape(X)[0]), sampling_threshold
        #    )
        #    X = X[rand_idx]

        return pairwise_distances(
            X, Y, metric=self.DISTANCE_METRIC
        )  # , n_jobs=multiprocessing.cpu_count())

    def calculate_state(
        self,
        X_query,
        STATE_ARGSECOND_PROBAS,
        STATE_DIFF_PROBAS,
        STATE_ARGTHIRD_PROBAS,
        STATE_DISTANCES_LAB,
        STATE_DISTANCES_UNLAB,
        STATE_PREDICTED_CLASS,
        model,
        labeled_index,
        train_unlabeled_X,
        TRUE_DISTANCES,
    ):
        possible_samples_probas = model.predict_proba(X_query)

        sorted_probas = -np.sort(-possible_samples_probas, axis=1)
        argmax_probas = sorted_probas[:, 0]

        state_list = argmax_probas.tolist()

        if STATE_ARGSECOND_PROBAS:
            argsecond_probas = sorted_probas[:, 1]
            state_list += argsecond_probas.tolist()
        if STATE_DIFF_PROBAS:
            state_list += (argmax_probas - sorted_probas[:, 1]).tolist()
        if STATE_ARGTHIRD_PROBAS:
            if np.shape(sorted_probas)[1] < 3:
                state_list += [0 for _ in range(0, len(X_query))]
            else:
                state_list += sorted_probas[:, 2].tolist()
        if STATE_PREDICTED_CLASS:
            state_list += model.predict(X_query).tolist()

        if TRUE_DISTANCES:
            if STATE_DISTANCES_LAB:
                # calculate average distance to labeled and average distance to unlabeled samples
                average_distance_labeled = (
                    np.sum(
                        self._pairwise_distances_subsampling(labeled_index, X_query),
                        axis=0,
                    )
                    / len(labeled_index)
                )
                state_list += average_distance_labeled.tolist()

            if STATE_DISTANCES_UNLAB:
                # calculate average distance to labeled and average distance to unlabeled samples
                average_distance_unlabeled = (
                    np.sum(
                        self._pairwise_distances_subsampling(
                            train_unlabeled_X, X_query
                        ),
                        axis=0,
                    )
                    / len(train_unlabeled_X)
                )
                state_list += average_distance_unlabeled.tolist()
        else:
            state_list += np.zeros(len(X_query) * 2).tolist()

        if self.STATE_INCLUDE_NR_FEATURES:
            state_list = [self.X.shape[1]] + state_list
        return np.array(state_list)

    def select(self, labeled_index, unlabeled_index, model, batch_size=1, **kwargs):
        labeled_index = self.X[labeled_index]

        max_sum = 0

        for i in range(0, kwargs["HYPER_SAMPLING_SEARCH_ITERATIONS"]):

            # if there are less than 20 samples left we  still have to show the ann 20 samples because of the fixed input size
            # so we just "pad" it with random indices
            if len(unlabeled_index) <= 20:
                padding = np.random.choice(unlabeled_index, 20 - len(unlabeled_index))
                possible_samples_indices = np.concatenate((unlabeled_index, padding))
                break
            random_sample_index = unlabeled_index.random_sample(
                20
            )  # self._fast_random_choice_list(unlabeled_index, 20)

            random_sample = self.X[random_sample_index]
            # calculate distance to each other
            total_distance = np.sum(
                self._pairwise_distances_subsampling(random_sample, random_sample)
            )

            total_distance += np.sum(
                self._pairwise_distances_subsampling(labeled_index, random_sample)
            )
            if total_distance > max_sum:
                max_sum = total_distance
                possible_samples_indices = random_sample_index

        X_query = self.X[possible_samples_indices]

        X_state = self.calculate_state(
            X_query,
            STATE_ARGSECOND_PROBAS=True,
            STATE_DIFF_PROBAS=False,
            STATE_ARGTHIRD_PROBAS=True,
            STATE_DISTANCES_LAB=True,
            STATE_DISTANCES_UNLAB=True,
            STATE_PREDICTED_CLASS=False,
            model=model,
            labeled_index=labeled_index,
            train_unlabeled_X=self.X[unlabeled_index],
            TRUE_DISTANCES=kwargs["TRUE_DISTANCES"],
        )
        X_state = np.reshape(X_state, (1, len(X_state)))
        Y_pred = self.sampling_classifier.predict(X_state)
        sorting = Y_pred
        # use the optimal values
        zero_to_one_values_and_index = list(zip(sorting, possible_samples_indices))
        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        return [v for k, v in ordered_list_of_possible_sample_indices[:batch_size]]


class ANNQueryBatch(ANNQuerySingle):
    STATE_UNCERTAINTIES = True
    STATE_DISTANCES = True
    STATE_DISTANCES_LAB = False
    STATE_PREDICTED_UNITY = False
    NR_QUERIES_PER_ITERATION = BATCH_SIZE
    # INITIAL_BATCH_SAMPLING_ARG = 750
    #  INITIAL_BATCH_SAMPLING_METHOD = "hybrid"
    #  INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST = 0.4
    INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB = 0
    INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY = 0
    #  INITIAL_BATCH_SAMPLING_HYBRID_UNCERT = 0.4

    def __init__(self, X=None, Y=None, **kwargs):
        super().__init__(X, Y, **kwargs)
        self.N_CLASSES = len(np.unique(self.Y))
        self.DISTANCE_METRIC = kwargs["DISTANCE_METRIC"]
        self.STATE_INCLUDE_NR_FEATURES = kwargs["STATE_INCLUDE_NR_FEATURES"]
        self.INITIAL_BATCH_SAMPLING_ARG = kwargs["INITIAL_BATCH_SAMPLING_ARG"]
        self.INITIAL_BATCH_SAMPLING_HYBRID_UNCERT = kwargs[
            "INITIAL_BATCH_SAMPLING_HYBRID_UNCERT"
        ]
        self.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST = kwargs[
            "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST"
        ]
        self.INITIAL_BATCH_SAMPLING_METHOD = kwargs["INITIAL_BATCH_SAMPLING_METHOD"]

    def _calculate_furthest_metric(self, batch_indices):
        return np.sum(
            pairwise_distances(self.X[batch_indices], metric=self.DISTANCE_METRIC)
        )

    def _calculate_furthest_lab_metric(self, batch_indices):
        return np.sum(
            pairwise_distances(
                self.X[batch_indices],
                self.X[self.labeled_index],
                metric=self.DISTANCE_METRIC,
            )
        )

    def _calculate_uncertainty_metric(self, batch_indices):
        Y_proba = self.Y_probas[batch_indices]
        margin = np.partition(-Y_proba, 1, axis=1)
        return np.sum(-np.abs(margin[:, 0] - margin[:, 1]))

    def _calculate_predicted_unity(self, unlabeled_sample_indices):
        Y_pred = self.Y_pred[unlabeled_sample_indices]
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
        return disagreement_score

    def _get_normalized_unity_encoding_mapping(self):
        # adopted from https://stackoverflow.com/a/44209393
        def partitions(n, I=1):
            yield (n,)
            for i in range(I, n // 2 + 1):
                for p in partitions(n - i, i):
                    yield (i,) + p

        N_CLASSES = self.N_CLASSES

        if N_CLASSES >= BATCH_SIZE:
            N_CLASSES = BATCH_SIZE
        possible_lengths = set()

        for possible_partition in partitions(BATCH_SIZE):
            if len(possible_partition) <= N_CLASSES:
                possible_lengths.add(
                    sum(
                        [
                            c * u
                            for c, u in zip(
                                sorted(possible_partition, reverse=True),
                                range(1, len(possible_partition) + 1),
                            )
                        ]
                    )
                )
        mapping = {}
        for i, possible_length in enumerate(sorted(possible_lengths)):
            mapping[possible_length] = i / (len(possible_lengths) - 1)
        return mapping

    def calculate_state(self, batch_indices, model):
        state_list = []
        if self.STATE_UNCERTAINTIES:
            # normalize by batch size
            state_list += [
                (self.NR_QUERIES_PER_ITERATION + self._calculate_uncertainty_metric(a))
                / self.NR_QUERIES_PER_ITERATION
                for a in batch_indices
            ]

        if self.STATE_DISTANCES:
            # normalize based on the assumption, that the whole vector space got first normalized to -1 to +1, then we can calculate the maximum possible distance like this:

            if self.DISTANCE_METRIC == "euclidean":
                normalization_denominator = (
                    2 * math.sqrt(np.shape(self.X)[1]) * self.NR_QUERIES_PER_ITERATION
                )
            elif self.DISTANCE_METRIC == "cosine":
                normalization_denominator = self.NR_QUERIES_PER_ITERATION

            state_list += [
                self._calculate_furthest_metric(a) / normalization_denominator
                for a in batch_indices
            ]
        if self.STATE_DISTANCES_LAB:
            if self.DISTANCE_METRIC == "euclidean":
                normalization_denominator = (
                    2 * math.sqrt(np.shape(self.X)[1]) * self.NR_QUERIES_PER_ITERATION
                )
            elif self.DISTANCE_METRIC == "cosine":
                normalization_denominator = self.NR_QUERIES_PER_ITERATION

            state_list += [
                self._calculate_furthest_lab_metric(a) / normalization_denominator
                for a in batch_indices
            ]
        if self.STATE_PREDICTED_UNITY:
            pred_unity_mapping = self._get_normalized_unity_encoding_mapping()
            # normalize in a super complicated fashion due to the encoding
            state_list += [
                pred_unity_mapping[self._calculate_predicted_unity(a)]
                for a in batch_indices
            ]

        if self.STATE_INCLUDE_NR_FEATURES:
            state_list = [self.X.shape[1]] + state_list
        return np.array(state_list)

    def sample_unlabeled_X(
        self,
        SAMPLE_SIZE,
        INITIAL_BATCH_SAMPLING_METHOD,
        INITIAL_BATCH_SAMPLING_ARG,
        labeled_index,
        unlabeled_index,
    ):
        index_batches = []
        if len(unlabeled_index) <= self.NR_QUERIES_PER_ITERATION:
            padding = np.random.choice(
                unlabeled_index, self.NR_QUERIES_PER_ITERATION - len(unlabeled_index)
            )
            possible_samples_indices = np.concatenate((unlabeled_index, padding))
            index_batches = [
                np.array(possible_samples_indices) for _ in range(0, SAMPLE_SIZE)
            ]
        else:
            if INITIAL_BATCH_SAMPLING_METHOD == "random":
                for _ in range(0, SAMPLE_SIZE):
                    index_batches.append(
                        np.random.choice(
                            unlabeled_index,
                            size=self.NR_QUERIES_PER_ITERATION,
                            replace=False,
                        )
                    )
            elif (
                INITIAL_BATCH_SAMPLING_METHOD == "furthest"
                or INITIAL_BATCH_SAMPLING_METHOD == "furthest_lab"
                or INITIAL_BATCH_SAMPLING_METHOD == "uncertainty"
                or INITIAL_BATCH_SAMPLING_METHOD == "predicted_unity"
            ):
                possible_batches = [
                    np.random.choice(
                        unlabeled_index,
                        size=self.NR_QUERIES_PER_ITERATION,
                        replace=False,
                    )
                    for x in range(0, INITIAL_BATCH_SAMPLING_ARG)
                ]

                if INITIAL_BATCH_SAMPLING_METHOD == "furthest":
                    metric_function = self._calculate_furthest_metric
                elif INITIAL_BATCH_SAMPLING_METHOD == "furthest_lab":
                    metric_function = self._calculate_furthest_lab_metric
                elif INITIAL_BATCH_SAMPLING_METHOD == "uncertainty":
                    metric_function = self._calculate_uncertainty_metric
                elif INITIAL_BATCH_SAMPLING_METHOD == "predicted_unity":
                    metric_function = self._calculate_predicted_unity
                metric_values = [metric_function(a) for a in possible_batches]

                # take n samples based on the sorting metric, the rest randomly
                index_batches = [
                    x
                    for _, x in sorted(
                        zip(metric_values, possible_batches),
                        key=lambda t: t[0],
                        reverse=True,
                    )
                ][:SAMPLE_SIZE]
            elif INITIAL_BATCH_SAMPLING_METHOD == "hybrid":
                possible_batches = [
                    np.random.choice(
                        unlabeled_index,
                        size=self.NR_QUERIES_PER_ITERATION,
                        replace=False,
                    )
                    for x in range(0, INITIAL_BATCH_SAMPLING_ARG)
                ]

                furthest_index_batches = [
                    x
                    for _, x in sorted(
                        zip(
                            [
                                self._calculate_furthest_metric(a)
                                for a in possible_batches
                            ],
                            possible_batches,
                        ),
                        key=lambda t: t[0],
                        reverse=True,
                    )
                ][
                    : math.floor(
                        SAMPLE_SIZE * self.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST
                    )
                ]

                furthest_lab_index_batches = [
                    x
                    for _, x in sorted(
                        zip(
                            [
                                self._calculate_furthest_lab_metric(a)
                                for a in possible_batches
                            ],
                            possible_batches,
                        ),
                        key=lambda t: t[0],
                        reverse=True,
                    )
                ][
                    : math.floor(
                        SAMPLE_SIZE * self.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB
                    )
                ]

                uncertainty_index_batches = [
                    x
                    for _, x in sorted(
                        zip(
                            [
                                self._calculate_uncertainty_metric(a)
                                for a in possible_batches
                            ],
                            possible_batches,
                        ),
                        key=lambda t: t[0],
                        reverse=True,
                    )
                ][: math.floor(SAMPLE_SIZE * self.INITIAL_BATCH_SAMPLING_HYBRID_UNCERT)]

                predicted_unity_index_batches = [
                    x
                    for _, x in sorted(
                        zip(
                            [
                                self._calculate_predicted_unity(a)
                                for a in possible_batches
                            ],
                            possible_batches,
                        ),
                        key=lambda t: t[0],
                        reverse=True,
                    )
                ][
                    : math.floor(
                        SAMPLE_SIZE * self.INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY
                    )
                ]

                index_batches = [
                    tuple(i.tolist())
                    for i in (
                        furthest_index_batches
                        + furthest_lab_index_batches
                        + uncertainty_index_batches
                        + predicted_unity_index_batches
                    )
                ]
                index_batches = set(index_batches)

                # add some random batches as padding
                index_batches = [np.array(list(i)) for i in index_batches] + [
                    np.array(i)
                    for i in random.sample(
                        set([tuple(i.tolist()) for i in possible_batches]).difference(
                            index_batches
                        ),
                        SAMPLE_SIZE - len(index_batches),
                    )
                ]
            else:
                print(
                    "NON EXISTENT INITIAL_SAMPLING_METHOD: "
                    + INITIAL_BATCH_SAMPLING_METHOD
                )
                raise ()
        return index_batches

    def select(self, labeled_index, unlabeled_index, model, batch_size=1, **kwargs):
        self.labeled_index = labeled_index
        self.unlabeled_index = unlabeled_index
        self.clf = model
        self.Y_probas = self.clf.predict_proba(self.X)
        self.Y_pred = self.clf.predict(self.X)
        batch_indices = self.sample_unlabeled_X(
            self.sampling_classifier.n_outputs_,
            INITIAL_BATCH_SAMPLING_ARG=self.INITIAL_BATCH_SAMPLING_ARG,
            INITIAL_BATCH_SAMPLING_METHOD=self.INITIAL_BATCH_SAMPLING_METHOD,
            labeled_index=labeled_index,
            unlabeled_index=unlabeled_index,
        )
        X_state = self.calculate_state(batch_indices, model=model)

        X_state = np.reshape(X_state, (1, len(X_state)))
        Y_pred = self.sampling_classifier.predict(X_state)
        sorting = Y_pred.argmax()

        return batch_indices[sorting]


query_strategies = {
    #  (
    #      ANNQuerySingle,
    #      {
    #          "NN_BINARY_PATH": "../datasets/taurus_10_10/MORE_DATA/03_imital_trained_ann.model",
    #          "HYPER_SAMPLING_SEARCH_ITERATIONS": 100,
    #          "TRUE_DISTANCES": True,
    #      },
    #  ),
    #  (
    #      ANNQuerySingle,
    #      {
    #          "NN_BINARY_PATH": "../datasets/taurus_10_10/MORE_DATA/03_imital_trained_ann.model",
    #          "HYPER_SAMPLING_SEARCH_ITERATIONS": 100,
    #          "TRUE_DISTANCES": False,
    #      },
    #  ),
    35: (
        ANNQueryBatch,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/batch.pickle",
            "DISTANCE_METRIC": "cosine",
            "STATE_INCLUDE_NR_FEATURES": False,
            "INITIAL_BATCH_SAMPLING_ARG": 1,
            "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST": 0,
            "INITIAL_BATCH_SAMPLING_HYBRID_UNCERT": 0,
            "INITIAL_BATCH_SAMPLING_METHOD": "random",
        },
    ),
    34: (
        ANNQueryBatch,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/batch.pickle",
            "DISTANCE_METRIC": "euclidean",
            "STATE_INCLUDE_NR_FEATURES": False,
            "INITIAL_BATCH_SAMPLING_ARG": 1,
            "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST": 0,
            "INITIAL_BATCH_SAMPLING_HYBRID_UNCERT": 0,
            "INITIAL_BATCH_SAMPLING_METHOD": "random",
        },
    ),
    32: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/single_10.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 1,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "euclidean",
            "STATE_INCLUDE_NR_FEATURES": False,
        },
    ),
    33: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/single_10.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 1,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "cosine",
            "STATE_INCLUDE_NR_FEATURES": False,
        },
    ),
    24: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/single_10.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 10,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "cosine",
            "STATE_INCLUDE_NR_FEATURES": False,
        },
    ),
    23: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/single_10.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 10,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "cosine",
            "STATE_INCLUDE_NR_FEATURES": False,
        },
    ),
    22: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH
            + "/single_10_cos_nrf_100features.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 10,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "cosine",
            "STATE_INCLUDE_NR_FEATURES": True,
        },
    ),
    21: (
        ANNQueryBatch,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/batch_nrf.pickle",
            "DISTANCE_METRIC": "euclidean",
            "STATE_INCLUDE_NR_FEATURES": True,
        },
    ),
    20: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/single_10_nrf.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 10,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "euclidean",
            "STATE_INCLUDE_NR_FEATURES": True,
        },
    ),
    19: (
        ANNQueryBatch,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/batch_cos.pickle",
            "DISTANCE_METRIC": "cosine",
            "STATE_INCLUDE_NR_FEATURES": False,
        },
    ),
    18: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/single_10_cos.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 10,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "cosine",
            "STATE_INCLUDE_NR_FEATURES": False,
        },
    ),
    17: (
        ANNQueryBatch,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/batch_cos_nrf.pickle",
            "DISTANCE_METRIC": "cosine",
            "STATE_INCLUDE_NR_FEATURES": True,
        },
    ),
    16: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/single_10_cos_nrf.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 10,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "cosine",
            "STATE_INCLUDE_NR_FEATURES": True,
        },
    ),
    15: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/single_10.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 10,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "euclidean",
            "STATE_INCLUDE_NR_FEATURES": False,
        },
    ),
    14: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/single.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 200,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "euclidean",
            "STATE_INCLUDE_NR_FEATURES": False,
        },
    ),
    13: (
        ANNQueryBatch,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/batch.pickle",
            "DISTANCE_METRIC": "cosine",
            "STATE_INCLUDE_NR_FEATURES": False,
        },
    ),
    0: (
        ANNQuerySingle,
        {
            "NN_BINARY_PATH": config.OUTPUT_PATH + "/single.pickle",
            "HYPER_SAMPLING_SEARCH_ITERATIONS": 10,
            "TRUE_DISTANCES": True,
            "DISTANCE_METRIC": "euclidean",
            "STATE_INCLUDE_NR_FEATURES": False,
        },
    ),
    2: ("QueryInstanceQBC", {}),
    3: ("QueryInstanceUncertainty", {"measure": "least_confident"}),
    4: ("QueryInstanceUncertainty", {"measure": "margin"}),
    5: ("QueryInstanceUncertainty", {"measure": "entropy"}),
    6: ("QueryInstanceRandom", {}),
    7: ("QureyExpectedErrorReduction", {}),
    8: ("QueryInstanceGraphDensity", {}),
    9: ("QueryInstanceQUIRE", {}),
    # the following are only for db4701
    10: ("QueryInstanceLAL", {}),  # memory
    11: ("QueryInstanceBMDR", {}),  # cvxpy
    12: ("QueryInstanceSPAL", {}),  # cvxpy
    #  6: ("QueryInstanceUncertainty", {"measure": "distance_to_boundary"}),
}

shuffling = np.random.permutation(len(Y))
X = X[shuffling]
Y = Y[shuffling]

scaler = RobustScaler()
X = scaler.fit_transform(X)

# scale back to [0,1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

test_ratio = 0.5
indices = [i for i in range(0, len(Y))]
train_idx = indices[: math.floor(len(Y) * (1 - test_ratio))]
test_idx = indices[math.floor(len(Y) * (1 - test_ratio)) :]
unlabel_idx = train_idx.copy()
label_idx = []
#  print(Y)
#  print(y[train_idx])
for label in np.unique(Y):
    if label not in y[train_idx]:
        print(np.where(y[test_idx] == label))
    init_labeled_index = np.where(y[train_idx] == label)[0][0]
    label_idx.append(init_labeled_index)
    unlabel_idx.remove(init_labeled_index)

train_idx = [np.array(train_idx)]
test_idx = [np.array(test_idx)]
label_idx = [np.array(label_idx)]
unlabel_idx = [np.array(unlabel_idx)]


def run_parallel(query_strategy):
    print(query_strategy)

    al = AlExperiment(
        X,
        y,
        #  model=MLPClassifier(),
        model=RandomForestClassifier(n_jobs=multiprocessing.cpu_count()),
        stopping_criteria="num_of_queries",
        num_of_queries=dataset_load_function[2],
        stopping_value=dataset_load_function[2],
        batch_size=BATCH_SIZE,
        train_idx=train_idx,
        test_idx=test_idx,
        label_idx=label_idx,
        unlabel_idx=unlabel_idx,
    )

    al.set_query_strategy(
        strategy=query_strategy[0], **query_strategy[1]
    )  # , measure="least_confident")

    #  al.set_performance_metric("accuracy_score")
    al.set_performance_metric("f1_score")

    start = timer()
    al.start_query(multi_thread=False)
    end = timer()

    trained_model = al._model

    r = al.get_experiment_result()

    stateio = r[0]
    metric_values = []
    if stateio.initial_point is not None:
        metric_values.append(stateio.initial_point)
    for state in stateio:
        metric_values.append(state.get_value("performance"))
    f1_auc = auc([i for i in range(0, len(metric_values))], metric_values) / (
        len(metric_values) - 1
    )
    print(f1_auc)
    for r2 in r:
        res = r2.get_result()
        res["dataset_id"] = DATASET_ID
        res["strategy_id"] = str(STRATEGY_ID)
        res["dataset_random_seed"] = DATASET_RANDOM_SEED
        res["strategy"] = str(query_strategy[0]) + str(query_strategy[1])
        res["duration"] = end - start
        res["f1_auc"] = f1_auc
        res = {**res, **synthetic_creation_args}
        with open(config.OUTPUT_PATH + "/result.csv", "a") as f:
            w = csv.DictWriter(f, fieldnames=res.keys())
            if len(open(config.OUTPUT_PATH + "/result.csv").readlines()) == 0:
                print("write header")
                w.writeheader()
            w.writerow(res)


run_parallel(query_strategies[STRATEGY_ID])
#  for query_strategy in query_strategies:
#  run_parallel((query_strategy))

# with Parallel(n_jobs=config.N_JOBS, backend="threading") as parallel:
#    output = parallel(
#        delayed(run_parallel)(query_strategy) for query_strategy in query_strategies
#    )
