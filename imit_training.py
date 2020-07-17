import csv
from pathlib import Path
import pandas as pd
from collections import defaultdict
import datetime
import hashlib
import math
import operator
import threading
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from active_learning.al_cycle_wrapper import eval_al

#  import np.random.distributions as dists
from json_tricks import dumps
from sklearn.preprocessing import LabelEncoder

from active_learning.cluster_strategies import (
    DummyClusterStrategy,
    MostUncertainClusterStrategy,
    RandomClusterStrategy,
    RoundRobinClusterStrategy,
)
from active_learning.dataStorage import DataStorage
from active_learning.experiment_setup_lib import (
    calculate_global_score,
    conf_matrix_and_acc,
    get_param_distribution,
    init_logger,
)
from active_learning.sampling_strategies import (
    BoundaryPairSampler,
    RandomSampler,
    UncertaintySampler,
    OptimalForecastSampler,
    ImitationLearner,
)

from active_learning.weak_supervision import WeakCert, WeakClust


import random
import numpy as np
from active_learning.al_cycle_wrapper import train_and_eval_dataset
from active_learning.experiment_setup_lib import (
    standard_config,
    init_logger,
)
from fake_experiment_oracle import FakeExperimentOracle

config = standard_config(
    [
        (
            ["--SAMPLING"],
            {"help": "Possible values: uncertainty, random, committe, boundary",},
        ),
        (["--DATASET_NAME"], {"required": True,}),
        (
            ["--CLUSTER"],
            {
                "default": "dummy",
                "help": "Possible values: dummy, random, mostUncertain, roundRobin",
            },
        ),
        (["--NR_LEARNING_ITERATIONS"], {"type": int, "default": 150000}),
        (["--NR_QUERIES_PER_ITERATION"], {"type": int, "default": 150}),
        (["--START_SET_SIZE"], {"type": int, "default": 1}),
        (
            ["--MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS"],
            {"type": float, "default": 0.5},
        ),
        (
            ["--UNCERTAINTY_RECOMMENDATION_CERTAINTY_THRESHOLD"],
            {"type": float, "default": 0.9},
        ),
        (["--UNCERTAINTY_RECOMMENDATION_RATIO"], {"type": float, "default": 1 / 100}),
        (["--SNUBA_LITE_MINIMUM_HEURISTIC_ACCURACY"], {"type": float, "default": 0.9}),
        (
            ["--CLUSTER_RECOMMENDATION_MINIMUM_CLUSTER_UNITY_SIZE"],
            {"type": float, "default": 0.7},
        ),
        (
            ["--CLUSTER_RECOMMENDATION_RATIO_LABELED_UNLABELED"],
            {"type": float, "default": 0.9},
        ),
        (["--WITH_UNCERTAINTY_RECOMMENDATION"], {"action": "store_true"}),
        (["--WITH_CLUSTER_RECOMMENDATION"], {"action": "store_true"}),
        (["--WITH_SNUBA_LITE"], {"action": "store_true"}),
        (["--PLOT"], {"action": "store_true"}),
        (["--STOPPING_CRITERIA_UNCERTAINTY"], {"type": float, "default": 0.7}),
        (["--STOPPING_CRITERIA_ACC"], {"type": float, "default": 0.7}),
        (["--STOPPING_CRITERIA_STD"], {"type": float, "default": 0.7}),
        (
            ["--ALLOW_RECOMMENDATIONS_AFTER_STOP"],
            {"action": "store_true", "default": False},
        ),
        (["--OUTPUT_DIRECTORY"], {"default": "tmp/"}),
        (["--HYPER_SEARCH_TYPE"], {"default": "random"}),
        (["--USER_QUERY_BUDGET_LIMIT"], {"type": float, "default": 200}),
        (["--AMOUNT_OF_PEAKED_OBJECTS"], {"type": int, "default": 12}),
        (["--MAX_AMOUNT_OF_WS_PEAKS"], {"type": int, "default": 1}),
        (["--AMOUNT_OF_LEARN_ITERATIONS"], {"type": int, "default": 1}),
        (["--USE_OPTIMAL_ONLY"], {"action": "store_true"}),
    ]
)

init_logger("console")
for i in range(0, config.AMOUNT_OF_LEARN_ITERATIONS):

    if config.RANDOM_SEED == -2:
        config.RANDOM_SEED = random.randint(0, 2147483647)
        np.random.seed(config.RANDOM_SEED)
        random.seed(config.RANDOM_SEED)

    hyper_parameters = vars(config)

    print("Learn iteration {}".format(i))

    MAX_USED_N_SAMPLES = 1000
    N_SAMPLES = 1000
    N_FEATURES = random.randint(10, 100)
    N_INFORMATIVE, N_REDUNDANT, N_REPEATED = [
        int(N_FEATURES * i) for i in np.random.dirichlet(np.ones(3), size=1).tolist()[0]
    ]

    N_CLASSES = random.randint(2, 10)
    N_CLUSTERS_PER_CLASS = random.randint(
        1, min(max(1, int(2 ** N_INFORMATIVE / N_CLASSES)), 10)
    )

    if N_CLASSES * N_CLUSTERS_PER_CLASS > 2 ** N_INFORMATIVE:
        i -= 1
        continue

    WEIGHTS = np.random.dirichlet(np.ones(N_CLASSES), size=1).tolist()[
        0
    ]  # list of weights, len(WEIGHTS) = N_CLASSES, sum(WEIGHTS)=1
    FLIP_Y = (
        np.random.pareto(2.0) + 1
    ) * 0.01  # amount of noise, larger values make it harder
    CLASS_SEP = random.uniform(
        0, 10
    )  # larger values spread out the clusters and make it easier
    HYPERCUBE = True  # if false random polytope
    SCALE = 0.01  # features should be between 0 and 1 now

    synthetic_creation_args = {
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_informative": N_INFORMATIVE,
        "n_redundant": N_REDUNDANT,
        "n_repeated": N_REPEATED,
        "n_classes": N_CLASSES,
        "n_clusters_per_class": N_CLUSTERS_PER_CLASS,
        "weights": WEIGHTS,
        "flip_y": FLIP_Y,
        "class_sep": CLASS_SEP,
        "hypercube": HYPERCUBE,
        "scale": SCALE,
    }

    data_storage = DataStorage(
        hyper_parameters["RANDOM_SEED"],
        hyper_parameters["DATASET_NAME"],
        hyper_parameters["DATASETS_PATH"],
        hyper_parameters["START_SET_SIZE"],
        hyper_parameters["TEST_FRACTION"],
        hyper_parameters,
        **synthetic_creation_args
    )

    hyper_parameters["LEN_TRAIN_DATA"] = len(data_storage.train_unlabeled_Y) + len(
        data_storage.train_labeled_Y
    )
    cluster_strategy = DummyClusterStrategy()
    cluster_strategy.set_data_storage(data_storage, hyper_parameters["N_JOBS"])

    classifier = RandomForestClassifier(
        n_jobs=hyper_parameters["N_JOBS"], random_state=hyper_parameters["RANDOM_SEED"]
    )

    weak_supervision_label_sources = []

    oracle = FakeExperimentOracle()

    active_learner_params = {
        "data_storage": data_storage,
        "cluster_strategy": cluster_strategy,
        "N_JOBS": hyper_parameters["N_JOBS"],
        "RANDOM_SEED": hyper_parameters["RANDOM_SEED"],
        "NR_LEARNING_ITERATIONS": hyper_parameters["NR_LEARNING_ITERATIONS"],
        "NR_QUERIES_PER_ITERATION": hyper_parameters["NR_QUERIES_PER_ITERATION"],
        "oracle": oracle,
        "clf": classifier,
        "weak_supervision_label_sources": weak_supervision_label_sources,
    }

    active_learner = ImitationLearner(**active_learner_params)
    active_learner.set_amount_of_peaked_objects(
        hyper_parameters["AMOUNT_OF_PEAKED_OBJECTS"]
    )

    active_learner.init_sampling_classifier(
        hyper_parameters["OUTPUT_DIRECTORY"], hyper_parameters["USE_OPTIMAL_ONLY"]
    )
    active_learner.MAX_AMOUNT_OF_WS_PEAKS = hyper_parameters["MAX_AMOUNT_OF_WS_PEAKS"]

    start = timer()
    trained_active_clf_list, metrics_per_al_cycle = active_learner.learn(
        **hyper_parameters
    )
    end = timer()

    active_learner.save_nn_training_data(hyper_parameters["OUTPUT_DIRECTORY"])

eval_al(
    data_storage,
    trained_active_clf_list,
    end - start,
    metrics_per_al_cycle,
    active_learner,
    hyper_parameters,
)
