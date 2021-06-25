from math import ceil
import typing
from active_learning.merge_weak_supervision_label_strategies.BaseMergeWeakSupervisionLabelStrategy import (
    BaseMergeWeakSupervisionLabelStrategy,
)
from active_learning.datasets.uci import load_uci
import csv
import os
import sys
from joblib import Parallel, delayed, parallel_backend
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
from scipy.stats import uniform, randint, loguniform
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
        (["--N_TASKS"], {"type": int, "default": 1}),
    ],
    return_parser=False,
)


def run_ws_plus_al_experiment(
    DATASET: str,
    DATASET_RANDOM_GENERATION_SEED: int,
    LF_RANDOM_SEED: int,
    FRACTION_OF_LASTLY_AL_LABELLED_SAMPLES: int,
    AL_SAMPLES_WEIGHT: int,
    MERGE_WS_SAMPLES_STRATEGY: str,
    AMOUNT_OF_LFS: float,
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
        data_storage.true_Y[
            data_storage.unlabeled_mask[:AMOUNT_OF_SAMPLES_TO_INITIALLY_LABEL]
        ],
        "I",
    )

    learner.fit(
        data_storage.X[data_storage.labeled_mask],
        data_storage.Y_merged_final[data_storage.labeled_mask],
    )

    Y_true = data_storage.true_Y[data_storage.test_mask]
    Y_pred = learner.predict(data_storage.X[data_storage.test_mask])

    acc_initial = accuracy_score(Y_true, Y_pred)
    f1_initial = f1_score(Y_true, Y_pred, average="weighted")

    # 2. now generate some labels via WS
    ws_list: List[SyntheticLabelingFunctions] = [
        SyntheticLabelingFunctions(
            X=data_storage.X, Y=data_storage.true_Y, RANDOM_SEED=LF_RANDOM_SEED + i
        )
        for i in range(0, ceil(AMOUNT_OF_LFS))
    ]  # type: ignore
    ABSTAIN_THRESHOLDS = [ws.ABSTAIN_THRESHOLD for ws in ws_list]
    LF_CLASSIFIERS = [ws.LF_CLASSIFIER_NAME for ws in ws_list]
    AMOUNT_OF_LF_FEATURESSS = [ws.AMOUNT_OF_LF_FEATURESSS for ws in ws_list]
    synthetic_creation_args["ABSTAIN_THRESHOLDS"] = ABSTAIN_THRESHOLDS
    synthetic_creation_args["LF_CLASSIFIERS"] = LF_CLASSIFIERS
    synthetic_creation_args["AMOUNT_OF_LF_FEATURESSS"] = AMOUNT_OF_LF_FEATURESSS
    synthetic_creation_args["acc_WS"] = []
    synthetic_creation_args["f1_WS"] = []
    # calculate accuracies of ws_s
    for ws in ws_list:
        Y_true = data_storage.true_Y[data_storage.test_mask]
        Y_pred = ws.get_labels(data_storage.test_mask, data_storage, None)

        synthetic_creation_args["acc_WS"].append(accuracy_score(Y_true, Y_pred))
        synthetic_creation_args["f1_WS"].append(
            f1_score(Y_true, Y_pred, average="weighted")
        )

    mergeStrategy: BaseMergeWeakSupervisionLabelStrategy
    if MERGE_WS_SAMPLES_STRATEGY == "MajorityVoteLabelMergeStrategy":
        mergeStrategy = MajorityVoteLabelMergeStrategy()
    elif MERGE_WS_SAMPLES_STRATEGY == "SnorkelLabelMergeStrategy":
        mergeStrategy = SnorkelLabelMergeStrategy(
            cardinality=synthetic_creation_args["n_classes"],
            random_seed=DATASET_RANDOM_GENERATION_SEED,
        )
    elif MERGE_WS_SAMPLES_STRATEGY == "RandomLabelMergeStrategy":
        mergeStrategy = RandomLabelMergeStrategy()
    else:
        print("Misspelled Merge WS Labeling Strategy")
        exit(-1)
    data_storage.set_weak_supervisions(
        typing.cast(List[BaseWeakSupervision], ws_list), mergeStrategy
    )
    data_storage.generate_weak_labels(learner)

    learner = get_classifier("RF", random_state=DATASET_RANDOM_GENERATION_SEED)
    learner.fit(
        data_storage.X[data_storage.weakly_combined_mask],
        data_storage.Y_merged_final[data_storage.weakly_combined_mask],
    )
    Y_true = data_storage.true_Y[data_storage.test_mask]
    Y_pred = learner.predict(data_storage.X[data_storage.test_mask])
    acc_ws = accuracy_score(Y_true, Y_pred)
    f1_ws = f1_score(Y_true, Y_pred, average="weighted")

    # 3. now add some labels by AL
    AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES = ceil(
        len(data_storage.unlabeled_mask) * FRACTION_OF_LASTLY_AL_LABELLED_SAMPLES
    )

    al_selected_indices: IndiceMask
    acc_ws_and_al: Dict[str, float] = {}
    f1_ws_and_al: Dict[str, float] = {}
    acc_al_and_al: Dict[str, float] = {}
    f1_al_and_al: Dict[str, float] = {}
    original_data_storage = copy.deepcopy(data_storage)
    for AL_SAMPLING_STRATEGY in [
        "UncertaintyMaxMargin_no_ws",
        "UncertaintyMaxMargin_with_ws",
        "Random",
        "CoveredByLeastAmountOfLf",
        "ClassificationIsMostWrong",
        "GreatestDisagreement",
    ]:
        data_storage = copy.deepcopy(original_data_storage)
        if AL_SAMPLING_STRATEGY == "UncertaintyMaxMargin_no_ws":
            # select those n samples based on uncertainty max margin
            Y_temp_proba = learner.predict_proba(
                data_storage.X[data_storage.unlabeled_mask]
            )
            margin = np.partition(-Y_temp_proba, 1, axis=1)  # type: ignore
            result = -np.abs(margin[:, 0] - margin[:, 1])
            argsort = np.argsort(-result)  # type: ignore
            query_indices = data_storage.unlabeled_mask[argsort]
            al_selected_indices = query_indices[:AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES]

        elif AL_SAMPLING_STRATEGY == "UncertaintyMaxMargin_with_ws":
            learner.fit(
                data_storage.X[data_storage.weakly_combined_mask],
                data_storage.Y_merged_final[data_storage.weakly_combined_mask],
            )
            Y_temp_proba = learner.predict_proba(
                data_storage.X[data_storage.unlabeled_mask]
            )
            margin = np.partition(-Y_temp_proba, 1, axis=1)  # type: ignore
            result = -np.abs(margin[:, 0] - margin[:, 1])
            argsort = np.argsort(-result)  # type: ignore
            query_indices = data_storage.unlabeled_mask[argsort]
            al_selected_indices = query_indices[:AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES]
        elif AL_SAMPLING_STRATEGY == "Random":
            # randomly select n samples
            al_selected_indices = np.random.choice(
                data_storage.unlabeled_mask,
                size=AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES,
                replace=False,
            )
        elif AL_SAMPLING_STRATEGY == "CoveredByLeastAmountOfLf":
            # count for each sample how often -1 is present -> take the top-k samples
            order = {v: i for i, v in enumerate(data_storage.unlabeled_mask)}
            for i, weak_labels in zip(
                data_storage.unlabeled_mask, data_storage.ws_labels_list
            ):
                order[i] = np.count_nonzero(weak_labels == -1)  # type: ignore

            al_selected_indices = sorted(data_storage.unlabeled_mask, key=lambda x: order[x])[:AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES]  # type: ignore
        elif AL_SAMPLING_STRATEGY == "ClassificationIsMostWrong":
            # count for each samples how often the LFs are wrong, and choose then the top-k samples
            order = {v: i for i, v in enumerate(data_storage.unlabeled_mask)}
            for i, weak_labels in zip(
                data_storage.unlabeled_mask, data_storage.ws_labels_list
            ):
                order[i] = np.count_nonzero(weak_labels != data_storage.true_Y[i])  # type: ignore

            al_selected_indices = sorted(data_storage.unlabeled_mask, key=lambda x: order[x])[:AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES]  # type: ignore
        elif AL_SAMPLING_STRATEGY == "GreatestDisagreement":
            # count how many different labels I have per sample -> choose the top-k samples
            order = {v: i for i, v in enumerate(data_storage.unlabeled_mask)}
            for i, weak_labels in zip(
                data_storage.unlabeled_mask, data_storage.ws_labels_list
            ):
                order[i] = len(np.unique(weak_labels))
            al_selected_indices = sorted(data_storage.unlabeled_mask, key=lambda x: order[x])[:AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES]  # type: ignore
        else:
            print("AL_SAMPLING_STRATEGY unkown, exiting")
            exit(-1)

        data_storage.label_samples(
            al_selected_indices, data_storage.true_Y[al_selected_indices], "AL"
        )

        # 4. final evaluation
        weights = []
        for indice in data_storage.weakly_combined_mask:
            if indice in data_storage.labeled_mask:
                weights.append(AL_SAMPLES_WEIGHT)
            else:
                weights.append(1)

        learner = get_classifier("RF", random_state=DATASET_RANDOM_GENERATION_SEED)
        learner.fit(
            data_storage.X[data_storage.weakly_combined_mask],
            data_storage.Y_merged_final[data_storage.weakly_combined_mask],
            sample_weight=weights,  # type: ignore
        )
        Y_true = data_storage.true_Y[data_storage.test_mask]
        Y_pred = learner.predict(data_storage.X[data_storage.test_mask])
        acc_ws_and_al[AL_SAMPLING_STRATEGY] = accuracy_score(Y_true, Y_pred)
        f1_ws_and_al[AL_SAMPLING_STRATEGY] = f1_score(
            Y_true, Y_pred, average="weighted"
        )

        learner = get_classifier("RF", random_state=DATASET_RANDOM_GENERATION_SEED)
        learner.fit(
            data_storage.X[data_storage.labeled_mask],
            data_storage.Y_merged_final[data_storage.labeled_mask],
        )
        Y_true = data_storage.true_Y[data_storage.test_mask]
        Y_pred = learner.predict(data_storage.X[data_storage.test_mask])
        acc_al_and_al[AL_SAMPLING_STRATEGY] = accuracy_score(Y_true, Y_pred)
        f1_al_and_al[AL_SAMPLING_STRATEGY] = f1_score(
            Y_true, Y_pred, average="weighted"
        )

    synthetic_creation_args["f1_initial"] = f1_initial
    synthetic_creation_args["acc_initial"] = acc_initial
    synthetic_creation_args["f1_ws"] = f1_ws
    synthetic_creation_args["acc_ws"] = acc_ws
    for k, v in acc_ws_and_al.items():
        synthetic_creation_args["acc_ws_and_al_" + k] = v
        synthetic_creation_args["f1_ws_and_al_" + k] = f1_ws_and_al[k]
        synthetic_creation_args["acc_al_and_al_" + k] = acc_al_and_al[k]
        synthetic_creation_args["f1_al_and_al_" + k] = f1_al_and_al[k]
    synthetic_creation_args[
        "amount_of_initial_al_samples"
    ] = AMOUNT_OF_SAMPLES_TO_INITIALLY_LABEL
    synthetic_creation_args[
        "amount_of_lastly_al_samples"
    ] = AMOUNT_OF_LASTLY_AL_LABELLED_SAMPLES

    return synthetic_creation_args


if config.STAGE == "WORKLOAD":
    # create CSV containing the params to run the experiments on

    datasets = list(set([v[0] for v in dataset_id_mapping.values()]))
    datasets.remove("synthetic_euc_cos_test")
    for _ in range(1, 10):
        datasets.append("synthetic")

    param_grid = {
        "DATASET": datasets,
        "DATASET_RANDOM_GENERATION_SEED": randint(1, 1000000),
        "LF_RANDOM_SEED": randint(1, 1000000),
        "FRACTION_OF_LASTLY_AL_LABELLED_SAMPLES": uniform(0, 1),
        "AL_SAMPLES_WEIGHT": randint(1, 100),
        "MERGE_WS_SAMPLES_STRATEGY": [
            "MajorityVoteLabelMergeStrategy",
            "SnorkelLabelMergeStrategy",
            "RandomLabelMergeStrategy",
        ],
        "AMOUNT_OF_LFS": loguniform(1, 10),
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
    df = pd.read_csv(
        config.OUTPUT_PATH + "/workload.csv",
        header=0,
        nrows=config.JOB_ID + 1,
    )
    params = df.loc[config.JOB_ID]

    np.random.seed(config.JOB_ID)
    random.seed(config.JOB_ID)

    # print(params)

    result = run_ws_plus_al_experiment(**params)  # type: ignore
    result["JOB_ID"] = config.JOB_ID
    result.update(params.to_dict())
    # print(result)
    with open(config.OUTPUT_PATH + "/exp_results.csv", "a") as f:
        w = csv.DictWriter(f, fieldnames=result.keys())
        if len(open(config.OUTPUT_PATH + "/exp_results.csv").readlines()) == 0:
            print("write header")
            w.writeheader()
        w.writerow(result)
    exit(0)
elif config.STAGE == "MULTI_CORE_JOBS":

    def run_code(i):
        cli = (
            "python run_how_to_combine_ws_and_al_labels_experiment.py --STAGE JOB --OUTPUT_PATH "
            + config.OUTPUT_PATH
            + " --JOB_ID "
            + str(i)
        )
        print("#" * 100)
        print(i)
        print(cli)
        print("#" * 100)
        print("\n")
        os.system(cli)

    with parallel_backend("loky", n_jobs=-1):
        Parallel()(delayed(run_code)(i) for i in range(config.N_TASKS))

    pass
else:
    print("Beg your pardon?")
    exit(-1)
