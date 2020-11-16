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
from sklearn.metrics import jaccard_score

df = pd.read_csv("metric_test_10000.csv")

# step 1: convert numbers to top_n ranking
df.loc[:, df.columns != "source"] = df.loc[:, df.columns != "source"].apply(
    lambda x: np.argsort(x), axis=1
)
print(df)

# step 2: calculate correctnesses of rankings
TOP_N = 15
AMOUNT_OF_METRICS = 5

current_baseline = None


def get_top_n(df_series, TOP_N=TOP_N):
    df_series = df_series.drop("source")
    sorted_index = [int(x) for _, x in sorted(zip(df_series, df_series.index))]
    return sorted_index[:TOP_N]


def jaccard_comparison(baseline, to_compare):
    baseline = get_top_n(baseline)
    to_compare = get_top_n(to_compare)
    return jaccard_score(baseline, to_compare, average="weighted")


total_metric_values = {key: 0 for key in df.iloc[0:5]["source"]}

for i in range(0, math.ceil((len(df) / AMOUNT_OF_METRICS))):
    for j in range(i * AMOUNT_OF_METRICS, (i + 1) * AMOUNT_OF_METRICS):
        if j == len(df):
            continue

        if j % AMOUNT_OF_METRICS == 0:
            current_baseline = df.iloc[j]
        else:
            jac = jaccard_comparison(current_baseline, df.iloc[j])
            total_metric_values[df.iloc[j]["source"]] += jac
            #  if jac > 0:
            #  print(df.iloc[j]["source"])

print(total_metric_values)
