import os
from active_learning.merge_weak_supervision_label_strategies import (
    RandomLabelMergeStrategy,
    SnorkelLabelMergeStrategy,
    MajorityVoteLabelMergeStrategy,
)
from active_learning.learner.standard import Learner, get_classifier
from active_learning.weak_supervision.SelfTraining import SelfTraining
import argparse
import numpy as np
import copy
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score
from timeit import default_timer as timer
from typing import List
from active_learning.config import get_active_config
from active_learning.dataStorage import DataStorage
from active_learning.datasets import load_synthetic
from active_learning.logger import init_logger
from collections import Counter
from sklearn.model_selection import ParameterSampler

from active_learning.weak_supervision import SyntheticLabelingFunctions
from active_learning.weak_supervision.BaseWeakSupervision import BaseWeakSupervision

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform, truncnorm, randint

sns.set_theme(style="whitegrid")

config: argparse.Namespace = get_active_config(  # type: ignore
    [
        (["--STAGE"], {}),
        (["--JOB_ID"], {"type": int, "default": -1}),
        (["--WORKLOAD_AMOUNT"], {"type": int, "default": 100}),
    ],
    return_parser=False,
)

if config.STAGE == "WORKLOAD":
    # create CSV containing the params to run the experiments on
    """n_samples: int,
    n_classes: int,
    n_features: int,
    dataset_hardness: float,
    amount_of_al_samples: int,
    al_samples_weight: int,
    merge_ws_sample_strategy: str,
    amount_of_lfs: int,
    al_sampling_strategy: str,
    lf_quality: str,"""
    param_grid = {
        "n_samples": randint(100, 5000),
        "n_classes": randint(1, 10),
        "n_features": randint(2, 100),
        "weights": uniform(0, 1),
        "flip_y": uniform(
            0, 0.2
        ),  # (np.random.pareto(2.0) + 1) * 0.01,  # type: ignore
        "class_sep": uniform(0, 10),
        "amount_of_al_samples": randint(5, 500),
        "al_samples_weight": randint(1, 100),
        "merge_ws_sample_strategy": [
            "MajorityVoteLabelMergeStrategy",
            "SnorkelLabelMergeStrategy",
            "RandomLabelMergeStrategy",
        ],
        "amount_of_lfs": randint(0, 50),
        "al_sampling_strategy": [
            "UncertaintyMaxMargin",
            "Random",
            "CovereyByLeastAmountOfLf",
            "ClassificationIsMostWrong",
            "GreatestDisagreement",
        ],
        "lf_quality": [1, 2, 3, 4, 5],
    }
    rng = np.random.RandomState(config.RANDOM_SEED)
    param_list = list(
        ParameterSampler(param_grid, n_iter=config.WORKLOAD_AMOUNT, random_state=rng)
    )
    df = pd.DataFrame(param_list)
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)

    df.to_csv(config.OUTPUT_PATH + "/workload.csv", index=False)
    print(df)
    print("Workload generated")
    exit(0)
elif config.STAGE == "JOB":
    # use the JOB_ID cli argumen to take the jobs from the workload csv
    config.RANDOM_SEED = random.randint(0, 2147483647)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    pass
else:
    print("Beg your pardon?")
    exit(-1)


def evaluate_and_print_prediction(Y_pred, Y_true, title):
    acc = accuracy_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred, average="weighted")
    c = Counter(Y_pred)

    return [
        title,
        acc,
        f1,
        c.most_common(1)[0][0],
        c.most_common(1)[0][1] / len(Y_pred),
    ]


def train_and_evaluate(title, original_data_storage, WEIGHTS=0, WS=True):
    data_storage = copy.deepcopy(original_data_storage)
    learner = get_classifier("RF", random_state=config.RANDOM_SEED)
    data_storage.generate_weak_labels(learner)

    if WEIGHTS != 0:
        weights = []
        for indice in data_storage.weakly_combined_mask:
            if indice in data_storage.labeled_mask:
                weights.append(WEIGHTS)
            else:
                weights.append(1)
    else:
        weights = None
    if WS:
        mask = data_storage.weakly_combined_mask
    else:
        mask = data_storage.labeled_mask

    learner.fit(
        data_storage.X[mask],
        data_storage.Y_merged_final[mask],
        sample_weight=weights,  # type: ignore
    )
    Y_pred = learner.predict(data_storage.X[data_storage.test_mask])

    Y_true = data_storage.exp_Y[data_storage.test_mask]

    return evaluate_and_print_prediction(Y_pred, Y_true, title)


def test_one_labeled_set(original_data_storage, label_strategy="random", param=5):
    data_storage = copy.deepcopy(original_data_storage)

    if label_strategy == "random":
        random_sample_ids = np.random.choice(
            data_storage.unlabeled_mask,
            size=param,
            replace=False,
        )

        data_storage.label_samples(
            random_sample_ids, data_storage.exp_Y[random_sample_ids], "AL"
        )

    return [
        train_and_evaluate("RF No WS", data_storage, WS=False)
        + [label_strategy, param],
        train_and_evaluate("RF No Weights", data_storage) + [label_strategy, param],
        train_and_evaluate("RF Weights 10", data_storage, WEIGHTS=10)
        + [label_strategy, param],
        train_and_evaluate("RF Weights 50", data_storage, WEIGHTS=50)
        + [label_strategy, param],
        train_and_evaluate("RF Weights 100", data_storage, WEIGHTS=100)
        + [label_strategy, param],
        train_and_evaluate("RF Weights 1000", data_storage, WEIGHTS=1000)
        + [label_strategy, param],
    ]


"""
- Anzahl der Samples
- Prozent initial gelabelter Daten bevor WS dazu kommt
- Methode um verschiedene LFs zusammenzuführen (random, majority, snorkel)
- AL Samples Weight
- anzahl an LFs
- AL query strategy (random, mm, labels who are covered by the least amount of LFs, samples where LF classification is the most wrong, greatest disagremment among lfs)
- WS LF mit coverage + abstain betrachten
- qualität der LFs (viele schlechte, viele mittelmäßige, viele wirklich gute, etc.)
- verschiedene datasets (inkl. all ihrer Parameter)
"""


def run_ws_plus_al_experiment(
    n_samples: int,
    n_classes: int,
    n_features: int,
    dataset_hardness: float,
    amount_of_al_samples: int,
    al_samples_weight: int,
    merge_ws_sample_strategy: str,
    amount_of_lfs: int,
    al_sampling_strategy: str,
    lf_quality: str,
):
    return None
