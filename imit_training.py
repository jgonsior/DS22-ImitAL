import os
import random
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from active_learning.al_cycle_wrapper import eval_al
from active_learning.cluster_strategies import DummyClusterStrategy
from active_learning.dataStorage import DataStorage
from active_learning.experiment_setup_lib import (
    get_active_config,
    init_logger,
    get_classifier,
)
from active_learning.experiment_setup_lib import log_it
from active_learning.sampling_strategies import ImitationLearner, ImitationBatchLearner
from fake_experiment_oracle import FakeExperimentOracle

#  import np.random.distributions as dists

config = get_active_config()

if not os.path.isfile(config.OUTPUT_DIRECTORY + "/states.csv"):
    if config.SAMPLING == "single":
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
                str(i) + "_proba_diff"
                for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
            ]
        if config.STATE_ARGTHIRD_PROBAS:
            columns += [
                str(i) + "_proba_argthird"
                for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
            ]
        if config.STATE_PREDICTED_CLASS:
            columns += [
                str(i) + "_pred_class"
                for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
            ]
        if config.STATE_DISTANCES_LAB:
            columns += [
                str(i) + "_proba_avg_dist_lab"
                for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
            ]
        if config.STATE_DISTANCES_UNLAB:
            columns += [
                str(i) + "_proba_avg_dist_unlab"
                for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
            ]
    else:
        columns = []
        if config.STATE_UNCERTAINTIES:
            columns += [
                str(i) + "_avg_uncert"
                for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
            ]

        if config.STATE_DISTANCES:
            columns += [
                str(i) + "_avg_dist" for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
            ]
        if config.STATE_DISTANCES_LAB:
            columns += [
                str(i) + "_avg_dist_lab"
                for i in range(0, config.AMOUNT_OF_PEAKED_OBJECTS)
            ]
        if config.STATE_PREDICTED_UNITY:
            columns += [
                str(i) + "_avg_pred_unity"
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

config = vars(config)
for i in range(0, config["AMOUNT_OF_LEARN_ITERATIONS"]):

    if random_but_not_random:
        config["RANDOM_SEED"] = random.randint(0, 2147483647)
        np.random.seed(config.RANDOM_SEED)
        random.seed(config.RANDOM_SEED)

    log_it("Learn iteration {}".format(i))

    data_storage = DataStorage(
        df=None,
        **config,
    )

    if config["STOP_AFTER_MAXIMUM_ACCURACY_REACHED"]:
        # calculate maximum theoretical accuracy
        tmp_clf = get_classifier(
            config["CLASSIFIER"], random_state=config["RANDOM_SEED"]
        )

        tmp_clf.fit(
            data_storage.X[
                np.concatenate(
                    (data_storage.labeled_mask, data_storage.unlabeled_mask), axis=0
                )
            ],
            data_storage.Y[
                np.concatenate(
                    (data_storage.labeled_mask, data_storage.unlabeled_mask), axis=0
                )
            ],
        )
        tmp_Y_pred = tmp_clf.predict(data_storage.X[data_storage.test_mask])
        THEORETICALLY_BEST_ACHIEVABLE_ACCURACY = (
            accuracy_score(data_storage.Y[data_storage.test_mask], tmp_Y_pred) * 0.95
        )
        config[
            "THEORETICALLY_BEST_ACHIEVABLE_ACCURACY"
        ] = THEORETICALLY_BEST_ACHIEVABLE_ACCURACY

    config["LEN_TRAIN_DATA"] = len(data_storage.unlabeled_mask) + len(
        data_storage.labeled_mask
    )
    cluster_strategy = DummyClusterStrategy()
    cluster_strategy.set_data_storage(data_storage, config["N_JOBS"])
    classifier = get_classifier(
        config["CLASSIFIER"],
        n_jobs=config["N_JOBS"],
        random_state=config["RANDOM_SEED"],
    )

    weak_supervision_label_sources = []

    oracle = FakeExperimentOracle()

    active_learner_params = {
        "data_storage": data_storage,
        "cluster_strategy": cluster_strategy,
        "oracle": oracle,
        "clf": classifier,
        "weak_supervision_label_sources": weak_supervision_label_sources,
    }

    if config["SAMPLING"] == "single":
        active_learner = ImitationLearner(**active_learner_params, **config)
    elif config["SAMPLING"] == "batch":
        active_learner = ImitationBatchLearner(**active_learner_params, **config)

    start = timer()
    trained_active_clf_list, metrics_per_al_cycle = active_learner.learn()
    end = timer()

    active_learner.save_nn_training_data(config["OUTPUT_DIRECTORY"])

eval_al(
    data_storage,
    trained_active_clf_list,
    end - start,
    metrics_per_al_cycle,
    active_learner,
    config,
)
