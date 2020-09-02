from sklearn.metrics import accuracy_score
import os
import os
import random
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from active_learning.al_cycle_wrapper import eval_al
from active_learning.cluster_strategies import DummyClusterStrategy
from active_learning.dataStorage import DataStorage
from active_learning.experiment_setup_lib import (
    get_active_config,
    init_logger,
    get_classifier,
)
from active_learning.sampling_strategies import ImitationLearner
from fake_experiment_oracle import FakeExperimentOracle

#  import np.random.distributions as dists

config = get_active_config()

if not os.path.isfile(config.OUTPUT_DIRECTORY + "/states.csv"):
    columns = [
        str(i) + "_proba_argfirst" for i in range(config.AMOUNT_OF_PEAKED_OBJECTS)
    ]
    if config.STATE_ARGSECOND_PROBAS:
        columns += [
            str(i) + "_proba_argsecond"
            for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
        ]
    if config.STATE_DIFF_PROBAS:
        columns += [
            str(i) + "_proba_diff" for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
        ]
    if config.STATE_ARGTHIRD_PROBAS:
        columns += [
            str(i) + "_proba_argthird"
            for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
        ]
    if config.STATE_DISTANCES:
        columns += [
            str(i) + "_proba_avg_dist_lab"
            for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
        ]
        columns += [
            str(i) + "_proba_avg_dist_unlab"
            for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
        ]
    if config.STATE_LRU_AREAS_LIMIT > 0:
        columns += [
            str(i) + "_lru_dist" for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
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
        VARIABLE_DATASET=hyper_parameters["VARIABLE_DATASET"],
        NEW_SYNTHETIC_PARAMS=hyper_parameters["NEW_SYNTHETIC_PARAMS"],
        HYPERCUBE=hyper_parameters["HYPERCUBE"],
        AMOUNT_OF_FEATURES=hyper_parameters["AMOUNT_OF_FEATURES"],
        GENERATE_NOISE=hyper_parameters["GENERATE_NOISE"],
        #  hyper_parameters["START_SET_SIZE"],
        #  hyper_parameters["TEST_FRACTION"],
    )

    if hyper_parameters["STOP_AFTER_MAXIMUM_ACCURACY_REACHED"]:
        # calculate maximum theoretical accuracy
        tmp_clf = get_classifier(
            hyper_parameters["CLASSIFIER"], random_state=hyper_parameters["RANDOM_SEED"]
        )

        tmp_clf.fit(
            pd.concat([data_storage.train_unlabeled_X, data_storage.train_labeled_X]),
            data_storage.train_unlabeled_Y["label"].to_list()
            + data_storage.train_labeled_Y["label"].to_list(),
        )
        tmp_Y_pred = tmp_clf.predict(data_storage.test_X)
        THEORETICALLY_BEST_ACHIEVABLE_ACCURACY = (
            accuracy_score(data_storage.test_Y, tmp_Y_pred) * 0.95
        )
        hyper_parameters[
            "THEORETICALLY_BEST_ACHIEVABLE_ACCURACY"
        ] = THEORETICALLY_BEST_ACHIEVABLE_ACCURACY

    hyper_parameters["LEN_TRAIN_DATA"] = len(data_storage.train_unlabeled_Y) + len(
        data_storage.train_labeled_Y
    )
    cluster_strategy = DummyClusterStrategy()
    cluster_strategy.set_data_storage(data_storage, hyper_parameters["N_JOBS"])
    classifier = get_classifier(
        hyper_parameters["CLASSIFIER"],
        n_jobs=hyper_parameters["N_JOBS"],
        random_state=hyper_parameters["RANDOM_SEED"],
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
        hyper_parameters["AMOUNT_OF_PEAKED_OBJECTS"],
    )

    active_learner.init_sampling_classifier(
        DATA_PATH=hyper_parameters["OUTPUT_DIRECTORY"],
        CONVEX_HULL_SAMPLING=hyper_parameters["CONVEX_HULL_SAMPLING"],
        VARIANCE_BOUND=hyper_parameters["VARIANCE_BOUND"],
        STATE_DISTANCES=hyper_parameters["STATE_DISTANCES"],
        STATE_DIFF_PROBAS=hyper_parameters["STATE_DIFF_PROBAS"],
        STATE_ARGTHIRD_PROBAS=hyper_parameters["STATE_ARGTHIRD_PROBAS"],
        STATE_LRU_AREAS_LIMIT=hyper_parameters["STATE_LRU_AREAS_LIMIT"],
        STATE_ARGSECOND_PROBAS=hyper_parameters["STATE_ARGSECOND_PROBAS"],
        STATE_NO_LRU_WEIGHTS=hyper_parameters["STATE_NO_LRU_WEIGHTS"],
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
