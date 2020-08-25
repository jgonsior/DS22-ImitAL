import pandas as pd
import os
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
        (["--PLOT_EVOLUTION"], {"action": "store_true"}),
        (["--VARIABLE_INPUT_SIZE"], {"action": "store_true"}),
        (["--REPRESENTATIVE_FEATURES"], {"action": "store_true"}),
        (["--NEW_SYNTHETIC_PARAMS"], {"action": "store_true"}),
        (["--HYPERCUBE"], {"action": "store_true"}),
        (["--AMOUNT_OF_FEATURES"], {"type": int, "default": -1}),
    ]
)

if not os.path.isfile(config.OUTPUT_DIRECTORY + "/states.csv"):
    columns = [
        str(i) + "_proba_max" for i in range(config.AMOUNT_OF_PEAKED_OBJECTS)
    ] + [str(i) + "_proba_diff" for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)]

    if config.REPRESENTATIVE_FEATURES:
        columns += [
            str(i) + "_avg_dist_lab" for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
        ]
        columns += [
            str(i) + "_avg_dist_unlab"
            for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
        ]

    states = pd.DataFrame(data=None, columns=columns)

    optimal_policies = pd.DataFrame(
        data=None,
        columns=[
            str(i) + "_true_peaked_normalised_acc"
            for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
        ],
    )

    if not os.path.exists(config.OUTPUT_DIRECTORY):
        os.makedirs(config.OUTPUT_DIRECTORY)

    states.to_csv(config.OUTPUT_DIRECTORY + "/states.csv", index=False)
    optimal_policies.to_csv(config.OUTPUT_DIRECTORY + "/opt_pol.csv", index=False)

if config.RANDOM_SEED == -2:
    random_but_not_random = True
else:
    random_but_not_random = False

init_logger(config.LOG_FILE)
for i in range(0, config.AMOUNT_OF_LEARN_ITERATIONS):

    if random_but_not_random:
        config.RANDOM_SEED = random.randint(0, 2147483647)
        np.random.seed(config.RANDOM_SEED)
        random.seed(config.RANDOM_SEED)

    hyper_parameters = vars(config)

    print("Learn iteration {}".format(i))

    data_storage = DataStorage(
        RANDOM_SEED=hyper_parameters["RANDOM_SEED"],
        hyper_parameters=hyper_parameters,
        df=None,
        DATASET_NAME=hyper_parameters["DATASET_NAME"],
        DATASETS_PATH=hyper_parameters["DATASETS_PATH"],
        PLOT_EVOLUTION=hyper_parameters["PLOT_EVOLUTION"],
        VARIABLE_INPUT_SIZE=hyper_parameters["VARIABLE_INPUT_SIZE"],
        NEW_SYNTHETIC_PARAMS=hyper_parameters["NEW_SYNTHETIC_PARAMS"],
        HYPERCUBE=hyper_parameters["HYPERCUBE"],
        AMOUNT_OF_FEATURES=hyper_parameters["AMOUNT_OF_FEATURES"],
        #  hyper_parameters["START_SET_SIZE"],
        #  hyper_parameters["TEST_FRACTION"],
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
        hyper_parameters["OUTPUT_DIRECTORY"],
        hyper_parameters["REPRESENTATIVE_FEATURES"],
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
