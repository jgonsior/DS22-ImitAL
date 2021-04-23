import argparse
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
from typing import List

from active_learning.BaseOracle import BaseOracle
from active_learning.activeLearner import ActiveLearner
from active_learning.al_cycle_wrapper import eval_al
from active_learning.callbacks import MetricCallback, test_acc_metric, test_f1_metric
from active_learning.callbacks.PrintLoggingStatisticsCallback import (
    PrintLoggingStatisticsCallback,
)
from active_learning.config import get_active_config
from active_learning.dataStorage import DataStorage
from active_learning.datasets import load_synthetic
from active_learning.learner import get_classifier
from active_learning.logger import init_logger
from active_learning.merge_weak_supervision_label_strategies.MajorityVoteLabelMergeStrategy import (
    MajorityVoteLabelMergeStrategy,
)
from active_learning.oracles import FakeExperimentOracle
from active_learning.query_sampling_strategies import ImitationLearner
from active_learning.query_sampling_strategies.BatchStateEncoding import (
    TrainImitALBatch,
)
from active_learning.query_sampling_strategies.ImitationLearner import TrainImitALSingle

from active_learning.stopping_criterias import ALCyclesStoppingCriteria
from active_learning.weak_supervision import SyntheticLabelingFunctions
from active_learning.weak_supervision.BaseWeakSupervision import BaseWeakSupervision

config: argparse.Namespace = get_active_config()  # type: ignore

if not os.path.isfile(config.OUTPUT_DIRECTORY + "/01_state_encodings_X.csv"):
    if not config.BATCH_MODE:
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

    if config.STATE_INCLUDE_NR_FEATURES:
        columns = ["nr_features"] + columns

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

    states.to_csv(config.OUTPUT_DIRECTORY + "/01_state_encodings_X.csv", index=False)
    optimal_policies.to_csv(
        config.OUTPUT_DIRECTORY + "/01_expert_actions_Y.csv", index=False
    )

# -2 means that a true random seed is used, all other numbers use the provided CLI argument random_seed
if config.RANDOM_SEED == -2:
    random_but_not_random = True
else:
    random_but_not_random = False


init_logger(config.LOG_FILE)

if random_but_not_random:
    config.RANDOM_SEED = random.randint(0, 2147483647)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

df, synthetic_creation_args = load_synthetic(
    config.RANDOM_SEED,
    config.NEW_SYNTHETIC_PARAMS,
    config.VARIABLE_DATASET,
    config.AMOUNT_OF_FEATURES,
    config.HYPERCUBE,
    config.GENERATE_NOISE,
)

data_storage: DataStorage = DataStorage(df=df, TEST_FRACTION=config.TEST_FRACTION)

if config.STOP_AFTER_MAXIMUM_ACCURACY_REACHED:
    # calculate maximum theoretical accuracy
    tmp_clf = get_classifier(config.CLASSIFIER, random_state=config.RANDOM_SEED)

    tmp_clf.fit(
        data_storage.X[
            np.concatenate(
                (data_storage.labeled_mask, data_storage.unlabeled_mask), axis=0
            )
        ],
        data_storage.Y_merged_final[
            np.concatenate(
                (data_storage.labeled_mask, data_storage.unlabeled_mask), axis=0
            )
        ],
    )
    tmp_Y_pred = tmp_clf.predict(data_storage.X[data_storage.test_mask])
    THEORETICALLY_BEST_ACHIEVABLE_ACCURACY = (
        accuracy_score(data_storage.Y_merged_final[data_storage.test_mask], tmp_Y_pred)
        * 0.99
    )
    config.THEORETICALLY_BEST_ACHIEVABLE_ACCURACY = (
        THEORETICALLY_BEST_ACHIEVABLE_ACCURACY
    )

config.LEN_TRAIN_DATA = len(data_storage.unlabeled_mask) + len(
    data_storage.labeled_mask
)


oracle: BaseOracle = FakeExperimentOracle()  # type: ignore


if config.WS_MODE:
    ws_list: List[BaseWeakSupervision] = [
        SyntheticLabelingFunctions(X=data_storage.X, Y=data_storage.exp_Y)
        for i in range(0, config.AMOUNT_OF_SYNTHETIC_LABELLING_FUNCTIONS)
    ]  # type: ignore

    data_storage.set_weak_supervisions(ws_list, MajorityVoteLabelMergeStrategy())

if config.BATCH_MODE:
    print("ui" * 1000)
    samplingStrategy: ImitationLearner = TrainImitALBatch(
        PRE_SAMPLING_METHOD=config.PRE_SAMPLING_METHOD,
        PRE_SAMPLING_ARG=config.PRE_SAMPLING_ARG,
        AMOUNT_OF_PEAKED_OBJECTS=config.AMOUNT_OF_PEAKED_OBJECTS,
        # fixme later (depending if BATCH_MODE is still being pursued in future) -> these are not the BATCH state args
        STATE_ARGSECOND_PROBAS=config.STATE_ARGSECOND_PROBAS,
        STATE_ARGTHIRD_PROBAS=config.STATE_ARGTHIRD_PROBAS,
        STATE_DIFF_PROBAS=config.STATE_DIFF_PROBAS,
        STATE_PREDICTED_CLASS=config.STATE_PREDICTED_CLASS,
        STATE_DISTANCES_LAB=config.STATE_DISTANCES_LAB,
        STATE_DISTANCES_UNLAB=config.STATE_DISTANCES_UNLAB,
        STATE_INCLUDE_NR_FEATURES=config.STATE_INCLUDE_NR_FEATURES,
    )
else:

    samplingStrategy = TrainImitALSingle(
        PRE_SAMPLING_METHOD=config.PRE_SAMPLING_METHOD,
        PRE_SAMPLING_ARG=config.PRE_SAMPLING_ARG,
        AMOUNT_OF_PEAKED_OBJECTS=config.AMOUNT_OF_PEAKED_OBJECTS,
        STATE_ARGSECOND_PROBAS=config.STATE_ARGSECOND_PROBAS,
        STATE_ARGTHIRD_PROBAS=config.STATE_ARGTHIRD_PROBAS,
        STATE_DIFF_PROBAS=config.STATE_DIFF_PROBAS,
        STATE_PREDICTED_CLASS=config.STATE_PREDICTED_CLASS,
        STATE_DISTANCES_LAB=config.STATE_DISTANCES_LAB,
        STATE_DISTANCES_UNLAB=config.STATE_DISTANCES_UNLAB,
        STATE_INCLUDE_NR_FEATURES=config.STATE_INCLUDE_NR_FEATURES,
    )
callbacks = {
    "acc_test": MetricCallback(test_acc_metric),
    "f1_test": MetricCallback(test_f1_metric),
    "logging": PrintLoggingStatisticsCallback(),
}
active_learner_params = {
    "query_sampling_strategy": samplingStrategy,
    "data_storage": data_storage,
    "oracle": oracle,
    "learner": get_classifier(config.CLASSIFIER, random_state=config.RANDOM_SEED),
    "callbacks": callbacks,
    "stopping_criteria": ALCyclesStoppingCriteria(
        config.TOTAL_BUDGET / config.BATCH_SIZE
    ),
    "BATCH_SIZE": config.BATCH_SIZE,
    "WS_MODE": config.WS_MODE,
    "USE_WS_LABELS_CONTINOUSLY": config.USE_WS_LABELS_CONTINOUSLY,
}

active_learner = ActiveLearner(**active_learner_params)

start = timer()
active_learner.al_cycle()
end = timer()

samplingStrategy.save_nn_training_data(config.OUTPUT_DIRECTORY)

hyper_parameters = vars(config)
hyper_parameters["synthetic_creation_args"] = synthetic_creation_args

eval_al(
    data_storage,
    end - start,
    callbacks,
    hyper_parameters,
)
