from collections import defaultdict
import math
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    jaccard_score,
    ndcg_score,
    cohen_kappa_score,
    label_ranking_loss,
    label_ranking_average_precision_score,
    dcg_score,
)

from itertools import combinations, combinations_with_replacement, product

df = pd.read_csv("metric_hybrid.csv")
print("Done reading csv")


# step 2: calculate correctnesses of rankings
TOP_N = 20
AMOUNT_OF_METRICS = len(df.source.unique())
NR_BATCHES = df.shape[1] - 1
current_baseline = None


def _jac_score(a, b):
    return len(np.intersect1d(a, b)), 1 / len(np.union1d(a, b))


print(df[0:20])
print(AMOUNT_OF_METRICS)


def binarize_row(row):
    row_without_source = row[1:]
    top_k = row_without_source.argsort(axis=0)[:TOP_N].to_numpy()
    new_row = [1 if i in top_k else 0 for i in range(0, NR_BATCHES)]
    #  print(np.array([row[0]] + new_row))
    return pd.Series(np.array([row[0]] + new_row))


df = df.apply(binarize_row, axis=1)
print(df[0:20])
print(df.columns)
df.renname(columns={"": "source"}, inplace=True)
df.to_csv("metric_hybrid_binary.csv", index=False)
