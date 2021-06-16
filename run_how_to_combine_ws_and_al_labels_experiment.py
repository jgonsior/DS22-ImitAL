from math import ceil
from active_learning.merge_weak_supervision_label_strategies.BaseMergeWeakSupervisionLabelStrategy import (
    BaseMergeWeakSupervisionLabelStrategy,
)
from active_learning.datasets.uci import load_uci
import csv
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
from typing import Any, Dict, List, Tuple
from active_learning.config import get_active_config
from active_learning.dataStorage import DataStorage, IndiceMask
from active_learning.datasets import load_synthetic
from active_learning.logger import init_logger
from collections import Counter
from sklearn.model_selection import ParameterSampler
from sklearn.datasets import make_classification

from active_learning.weak_supervision import SyntheticLabelingFunctions
from active_learning.weak_supervision.BaseWeakSupervision import BaseWeakSupervision

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from imitLearningPipelineSharedCode import dataset_id_mapping
from active_learning.query_sampling_strategies import (
    UncertaintyQuerySampler,
    RandomQuerySampler,
)

sns.set_theme(style="whitegrid")

config: argparse.Namespace = get_active_config(  # type: ignore
    [
        (["--STAGE"], {}),
        (["--JOB_ID"], {"type": int, "default": -1}),
        (["--WORKLOAD_AMOUNT"], {"type": int, "default": 100}),
    ],
    return_parser=False,
)
config.DATASETS_PATH = "~/datasets"

"""
open problems: now I only have LFs of fixed size X
-> prior to that I had everything UP TO X --> maybe save that as new parameteres in the output??!!
-> and save in the result df also all the parameters from this experiment

are the lfs even used? (thresholds, everything seems to be -1 now??)

allow also more than one type of LF_classifiers -> categorical varibales aka "has_lf" "has_dt", "has_knn"

synthetic datasets sem to be the same???
"""


def run_ws_plus_al_experiment(
    DATASET: str,
    DATASET_RANDOM_GENERATION_SEED: int,
    FRACTION_OF_LASTLY_AL_LABELLED_SAMPLES: int,
    AL_SAMPLES_WEIGHT: int,
    MERGE_WS_SAMPLES_STRATEGY: str,
    AMOUNT_OF_LFS: int,
    AL_SAMPLING_STRATEGY: str,
    ABSTAIN_THRESHOLD: float,
    AMOUNT_OF_LF_FEATURES: int,
    LF_CLASSIFIER: str,
    FRACTION_OF_INITIALLY_LABELLED_SAMPLES: int,
) -> Dict[str, Any]:
    if DATASET == "synthetic":
        df, synthetic_creation_args = load_synthetic(
            DATASET_RANDOM_GENERATION_SEED,
        )
    else:
        df, synthetic_creation_args = load_uci(
            config.DATASETS_PATH, DATASET, DATASET_RANDOM_GENERATION_SEED
        )
    print("Loaded " + DATASET)
    print(synthetic_creation_args)

    data_storage: DataStorage = DataStorage(df=df, TEST_FRACTION=0.5)
    learner: Learner = get_classifier("RF", random_state=DATASET_RANDOM_GENERATION_SEED)

    # 1. initially label some data
    AMOUNT_OF_SAMPLES_TO_INITIALLY_LABEL: int = ceil(
        len(df) / 2 * FRACTION_OF_INITIALLY_LABELLED_SAMPLES
    )

    data_storage.label_samples(
        data_storage.unlabeled_mask[:AMOUNT_OF_SAMPLES_TO_INITIALLY_LABEL],
        data_storage.exp_Y[
            data_storage.unlabeled_mask[:AMOUNT_OF_SAMPLES_TO_INITIALLY_LABEL]
        ],
        "I",
    )

    learner.fit(
        data_storage.X[data_storage.labeled_mask],
        data_storage.Y_merged_final[data_storage.labeled_mask],
    )
    Y_true = data_storage.exp_Y[data_storage.test_mask]
    Y_pred = learner.predict(data_storage.X[data_storage.test_mask])
    acc_initial = accuracy_score(Y_true, Y_pred)
    f1_initial = f1_score(Y_true, Y_pred, average="weighted")

    # 2. now generate some labels via WS
    ws_list: List[BaseWeakSupervision] = [
        SyntheticLabelingFunctions(
            X=data_storage.X,
            Y=data_storage.exp_Y,
            ABSTAIN_THRESHOLD=ABSTAIN_THRESHOLD,
            AMOUNT_OF_LF_FEATURES=AMOUNT_OF_LF_FEATURES,
            LF_CLASSIFIER=LF_CLASSIFIER,
        )
        for _ in range(0, AMOUNT_OF_LFS)
    ]  # type: ignore

    mergeStrategy: BaseMergeWeakSupervisionLabelStrategy
    if MERGE_WS_SAMPLES_STRATEGY == "MajorityVoteLabelMergeStrategy":
        mergeStrategy = MajorityVoteLabelMergeStrategy()
    elif MERGE_WS_SAMPLES_STRATEGY == "SnorkelLabelMergeStrategy":
        mergeStrategy = SnorkelLabelMergeStrategy()
    elif MERGE_WS_SAMPLES_STRATEGY == "RandomLabelMergeStrategy":
        mergeStrategy = RandomLabelMergeStrategy()
    else:
        print("Misspelled Merge WS Labeling Strategy")
        exit(-1)
    data_storage.set_weak_supervisions(ws_list, mergeStrategy)
    data_storage.generate_weak_labels(learner)

    learner.fit(
        data_storage.X[data_storage.labeled_mask],
        data_storage.Y_merged_final[data_storage.labeled_mask],
    )
    Y_true = data_storage.exp_Y[data_storage.test_mask]
    Y_pred = learner.predict(data_storage.X[data_storage.test_mask])
    acc_ws = accuracy_score(Y_true, Y_pred)
    f1_ws = f1_score(Y_true, Y_pred, average="weighted")

    # 3. now add some labels by AL
    AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES = ceil(len(data_storage.unlabeled_mask)*FRACTION_OF_LASTLY_AL_LABELLED_SAMPLES)

    al_selected_indices: IndiceMask
    if AL_SAMPLING_STRATEGY == "UncertaintyMaxMargin_no_ws":
        # select thos n samples based on uncertainty max margin
        # first: based on trained RF without WS
        # second variant based on RF trained using the weaklabelling functions
        sampling_strategy = UncertaintyQuerySampler()
        #al_sampled_ids =
    elif AL_SAMPLING_STRATEGY == "UncertaintyMaxMargin_with_ws":
        # select thos n samples based on uncertainty max margin
        # first: based on trained RF without WS
        # second variant based on RF trained using the weaklabelling functions
        sampling_strategy = UncertaintyQuerySampler()
    elif AL_SAMPLING_STRATEGY == "Random":
        # randomly select n samples
        al_selected_indices = np.random.choice(
            data_storage.unlabeled_mask,
            size=AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES,
            replace=False,
        )
    elif AL_SAMPLING_STRATEGY == "CoveredByLeastAmountOfLf":
        # count for each sample how often -1 is present -> take the top-k samples
        order = {v:i for i,v in enumerate(data_storage.unlabeled_mask)}
        for i, weak_labels in zip(data_storage.unlabeled_mask, data_storage.ws_labels_list):
            order[i] = np.count_nonzero(weak_labels == -1) # type: ignore

        al_selected_indices = sorted(data_storage.unlabeled_mask, key=lambda x:order[x])[:AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES]
    elif AL_SAMPLING_STRATEGY == "ClassificationIsMostWrong":
        al_selected_indices = []

        # count for each samples how often the LFs are wrong, and choose then the top-k samples
        sampling_strategy = ClassificationIsMostWrong()
    elif AL_SAMPLING_STRATEGY == "GreatestDisagreement":
        # count how many different labels I have per sample -> choose the top-k samples
        sampling_strategy = GreatestDisagreement()
    else:
        print("AL_SAMPLING_STRATEGY unkown, exiting")
        exit(-1)

    data_storage.label_samples(al_selected_indices, data_storage.exp_Y[al_selected_indices], "AL")

    # 4. final evaluation
    weights = []
    for indice in data_storage.weakly_combined_mask:
        if indice in data_storage.labeled_mask:
            weights.append(AL_SAMPLES_WEIGHT)
        else:
            weights.append(1)

    learner.fit(
        data_storage.X[data_storage.weakly_combined_mask],
        data_storage.Y_merged_final[data_storage.weakly_combined_mask],
        sample_weight=weights,  # type: ignore
    )
    Y_true = data_storage.exp_Y[data_storage.test_mask]
    Y_pred = learner.predict(data_storage.X[data_storage.test_mask])
    acc_ws_and_al = accuracy_score(Y_true, Y_pred)
    f1_ws_and_al = f1_score(Y_true, Y_pred, average="weighted")

    synthetic_creation_args["f1_initial"] = f1_initial
    synthetic_creation_args["acc_initial"] = acc_initial
    synthetic_creation_args["f1_ws"] = f1_ws
    synthetic_creation_args["acc_ws"] = acc_ws
    synthetic_creation_args["f1_ws_and_al"] = f1_ws_and_al
    synthetic_creation_args["acc_ws_and_al"] = acc_ws_and_al
    synthetic_creation_args["amount_of_initial_al_samples"] = AMOUNT_OF_SAMPLES_TO_INITIALLY_LABEL
    synthetic_creation_args["amount_of_lastly_al_samples"] = AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES
    print(sorted(synthetic_creation_args))
    return synthetic_creation_args


if config.STAGE == "WORKLOAD":
    # create CSV containing the params to run the experiments on

    datasets = list(set([v[0] for v in dataset_id_mapping.values()]))
    datasets.remove("synthetic_euc_cos_test")

    param_grid = {
        "DATASET": datasets,
        "DATASET_RANDOM_GENERATION_SEED": randint(1, 1000000),
        "FRACTION_OF_LASTLY_AL_LABELLED_SAMPLES": uniform(0,1),
        "AL_SAMPLES_WEIGHT": randint(1, 100),
        "MERGE_WS_SAMPLES_STRATEGY": [
            "MajorityVoteLabelMergeStrategy",
            "SnorkelLabelMergeStrategy",
            "RandomLabelMergeStrategy",
        ],
        "AMOUNT_OF_LFS": randint(0, 10),
        "AL_SAMPLING_STRATEGY": [
            "UncertaintyMaxMargin_no_ws",
            "UncertaintyMaxMargin_with_ws",
            "Random",
            "CoveredByLeastAmountOfLf",
            "ClassificationIsMostWrong",
            "GreatestDisagreement",
        ],
        "ABSTAIN_THRESHOLD": uniform(0, 1),
        "AMOUNT_OF_LF_FEATURES": uniform(0, 1),
        "LF_CLASSIFIER": ["dt", "lr", "knn"],
        "FRACTION_OF_INITIALLY_LABELLED_SAMPLES": uniform(0, 1),
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
    # use the JOB_ID cli argument to take the jobs from the workload csv
    config.RANDOM_SEED = random.randint(0, 2147483647)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    df = pd.read_csv(
        config.OUTPUT_PATH + "/workload.csv",
        header=0,
        nrows=config.JOB_ID + 1,
    )
    params = df.loc[config.JOB_ID]

    print(params)

    result = run_ws_plus_al_experiment(**params)  # type: ignore
    result.update(params.to_dict())
    with open(config.OUTPUT_PATH + "/exp_results.csv", "a") as f:
        w = csv.DictWriter(f, fieldnames=result.keys())
        if len(open(config.OUTPUT_PATH + "/exp_results.csv").readlines()) == 0:
            print("write header")
            w.writeheader()
        w.writerow(result)
    exit(0)
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
