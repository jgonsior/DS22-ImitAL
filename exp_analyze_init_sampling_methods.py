from pprint import pprint
import copy
from joblib import Parallel, delayed, parallel_backend
from numba import jit
import random
import math
import pandas as pd
import numpy as np
import timeit
from sklearn.datasets import make_classification
from active_learning.dataStorage import DataStorage
from active_learning.experiment_setup_lib import init_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances, accuracy_score
import multiprocessing
from active_learning.sampling_strategies.learnedBaseBatchSampling import (
    LearnedBaseBatchSampling,
)
import math
from sklearn.metrics import (
    jaccard_score,
    ndcg_score,
    cohen_kappa_score,
    label_ranking_loss,
    label_ranking_average_precision_score,
    dcg_score,
)


df = pd.read_csv("metric_test_10000.csv")

# step 2: calculate correctnesses of rankings
TOP_N = 5
AMOUNT_OF_METRICS = 5
NR_BATCHES = 50
current_baseline = None


def _binarize_targets(df, TOP_N=5):
    df = df.to_frame()
    df["threshold"] = [
        np.sort(np.reshape(df.values, NR_BATCHES))[-TOP_N : -(TOP_N - 1)]
        for _ in range(0, NR_BATCHES)
    ]

    for column_name in df.columns:
        if column_name == "threshold":
            continue
        df[column_name].loc[df[column_name] < df.threshold] = 0
        df[column_name].loc[df[column_name] >= df.threshold] = 1
    del df["threshold"]
    return np.reshape(df.to_numpy(), (1, NR_BATCHES))


def get_top_n(df_series, TOP_N=TOP_N):
    sorted_index = [int(x) for _, x in sorted(zip(df_series, df_series.index))]
    return sorted_index[:TOP_N]


def jaccard_comparison(baseline, to_compare):
    return jaccard_score(baseline, to_compare, average="weighted")


metrics = [
    jaccard_score,
    ndcg_score,
    "ndcg_score_top_k",
    cohen_kappa_score,
    label_ranking_loss,
    label_ranking_average_precision_score,
    dcg_score,
]

evaluation = {}

for metric in metrics:
    evaluation[metric] = {key: 0 for key in df.iloc[0:5]["source"]}
    evaluation[metric]["random"] = 0
    evaluation[metric]["maximum"] = 0

for i in range(0, math.ceil((len(df) / AMOUNT_OF_METRICS))):
    for j in range(i * AMOUNT_OF_METRICS, (i + 1) * AMOUNT_OF_METRICS):
        if j == len(df):
            continue

        if j % AMOUNT_OF_METRICS == 0:
            current_baseline = df.iloc[j].drop("source")
            df.loc[j, "source"] = "maximum"

        for metric in metrics:
            metric_function = metric
            kwargs = {}
            if metric == jaccard_score or metric == cohen_kappa_score:
                # top_n rankings
                baseline = get_top_n(current_baseline.argsort(axis=0))
                to_compare = get_top_n(df.iloc[j].drop("source"))
                if metric_function == jaccard_score:
                    kwargs = {"average": "weighted"}
                else:
                    baseline = sorted(baseline)
                    to_compare = sorted(to_compare)
            elif (
                metric == ndcg_score
                or metric == "ndcg_score_top_k"
                or metric == dcg_score
            ):
                # true values
                baseline = np.reshape(current_baseline.to_numpy(), (1, NR_BATCHES))
                to_compare = np.reshape(
                    df.iloc[j].drop("source").to_numpy(), (1, NR_BATCHES)
                )
                if metric == "ndcg_score_top_k" or metric == dcg_score:
                    kwargs = {"k": TOP_N}
                    metric_function = ndcg_score
            elif metric == label_ranking_loss:
                # 000111000 arrays
                baseline = _binarize_targets(current_baseline, TOP_N=TOP_N)
                to_compare = _binarize_targets(df.iloc[j].drop("source"), TOP_N=TOP_N)
            elif metric == label_ranking_average_precision_score:
                baseline = _binarize_targets(current_baseline, TOP_N=TOP_N)
                to_compare = np.reshape(
                    df.iloc[j].drop("source").to_numpy(), (1, NR_BATCHES)
                )

            evaluation[metric][df.iloc[j]["source"]] += metric_function(
                baseline, to_compare, **kwargs
            )

# per metric a bar chart
pprint(evaluation)
